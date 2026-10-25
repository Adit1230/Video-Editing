"""Bounded source-attention storage for image UNets and native video transformers."""
from pathlib import Path
import math
import hashlib
import re
import shutil
import warnings
import torch
import torch.nn.functional as F
from engine import align_tokens
from cache_manager import atomic_save


def mask_grid(mask, grid):
    """[frames,1,H,W] -> [latent_frames,h,w], conservatively pooling video time."""
    f, h, w = grid
    x = mask.float().permute(1, 0, 2, 3).unsqueeze(0)  # 1,1,F,H,W
    if x.shape[2] != f and x.shape[2] == 1 + 4*(f-1):
        groups = [x[:, :, :1]]
        groups += [x[:, :, 1+4*i:1+4*(i+1)].amax(2, keepdim=True) for i in range(f-1)]
        x = torch.cat(groups, 2)
    return F.interpolate(x, size=(f,h,w), mode='trilinear', align_corners=False)[0,0]


class SourceAttention:
    def __init__(self, root, steps, config, mask=None):
        self.root, self.steps, self.c = Path(root), steps, config
        self.root.mkdir(parents=True, exist_ok=True)
        self.mask = mask
        self.enabled = config.get('cache_attention', True)
        self.mode, self.step, self.snapshot = 'off', 0, {}
        self.disk_manager = None
        self.snapshot_bytes = 0
        self.mapper = self.matched = None
        self.tokens = []
        self.heat_mask = None
        self.grid = None
        self.stream_active=False
        self.stream_bytes=self.stream_ram_bytes=self.stream_budget=0
        self.stream_gpu_bytes=0
        self.gpu_cache_disabled=False
        self.gpu_cache_limit=None
        self._mask_cache={}
        self._alignment_cache={}
        self.image_query_limit=config.get('attention_max_queries',1024)
        self.native_query_limit=config.get('attention_max_queries',1024)
        self.stats = dict(stored_tensors=0, fused_self=0, fused_cross=0, cache_bytes=0,
                          stream_peak_bytes=0,stream_peak_ram_bytes=0,stream_spilled_bytes=0,
                          stream_peak_gpu_bytes=0,d2h_bytes=0,h2d_bytes=0,gpu_cache_oom_fallbacks=0)

    def align(self, source, target):
        self.mapper, self.matched = align_tokens(source, target)
        self._alignment_cache={}

    def begin(self, mode, step):
        if mode=='source':
            self.clear_stream()
            self.stream_active=True
            self.gpu_cache_disabled=self.gpu_cache_limit==0
            self.stream_budget=stream_ram_budget(self.c.get('stream_cache_ram_gib',0))
            mode='invert'
        self.mode, self.step, self.heat_mask = mode, step, None
        self._mask_cache={}
        if not self.stream_active:self.snapshot = {}
        if mode == 'edit' and self.enabled:
            if not self.stream_active:
                self.snapshot = torch.load(self.root/f'{self.steps-step-1:03d}.pt', map_location='cpu', weights_only=True)
            maps = [self.value(name).float() for name in self.snapshot if name.endswith('/heat')]
            if maps:
                device=next((value.device for value in maps if value.is_cuda),torch.device('cpu'))
                heat = torch.stack([value.to(device) for value in maps]).mean(0)  # F,1,H,W
                heat = F.max_pool2d(heat,3,1,1)
                self.heat_mask = (heat/heat.amax((-1,-2),keepdim=True).clamp_min(1e-8) > self.c.get('mask_threshold',.3)).float()

    def wants_image(self,cross):
        if not self.enabled or self.mode=='off':return False
        if not self.stream_active and self.mode=='invert':return True  # archive compatibility
        fraction=self.c['cross_replace' if cross else 'self_replace']
        return self.step<int(self.steps*fraction)

    def wants_heat(self):
        return (self.enabled and self.mode=='invert' and bool(self.tokens)
            and (not self.stream_active or self.c['mask_backend'] in ('attention','hybrid')))

    def alignment(self,device):
        key=str(device)
        if key not in self._alignment_cache:
            self._alignment_cache[key]=(self.mapper.to(device),self.matched.to(device))
        return self._alignment_cache[key]

    def gpu_room(self,value):
        if (not self.stream_active or self.c.get('stream_cache_device','auto')=='cpu'
            or self.gpu_cache_disabled or not value.is_cuda):return 0
        free,_=torch.cuda.mem_get_info(value.device)
        # Cached allocator blocks can be reused without another driver allocation.
        free+=max(0,torch.cuda.memory_reserved(value.device)-torch.cuda.memory_allocated(value.device))
        room=max(0,free-int(self.c.get('stream_gpu_reserve_gib',4.)*2**30))
        cap=self.c.get('stream_cache_gpu_gib',0.)
        limit=int(cap*2**30) if cap else None
        if self.gpu_cache_limit is not None:limit=self.gpu_cache_limit if limit is None else min(limit,self.gpu_cache_limit)
        return min(room,max(0,limit-self.stream_gpu_bytes)) if limit is not None else room

    def cpu_copy(self,value):
        if value.is_cuda:self.stats['d2h_bytes']+=value.numel()*value.element_size()
        return value.detach().to('cpu',copy=True)

    def offload_gpu_cache(self):
        # Preserve all maps for a deterministic retry, including maps already
        # consumed by a partially completed target forward. No head/query pruning.
        self.gpu_cache_disabled=True
        self.gpu_cache_limit=max(0,self.stream_gpu_bytes-2*2**30)
        self.stats['gpu_cache_limit_after_oom_bytes']=self.gpu_cache_limit
        self.stats['gpu_cache_oom_fallbacks']+=1
        for name,value in list(self.snapshot.items()):
            if torch.is_tensor(value) and value.is_cuda:
                size=value.numel()*value.element_size()
                self.stream_gpu_bytes-=size
                self._store_cpu(name,value,size)
        self.stats['stream_peak_ram_bytes']=max(self.stats['stream_peak_ram_bytes'],self.stream_ram_bytes)
        self._mask_cache={}

    def _store_cpu(self,name,value,size):
        if self.stream_active and self.stream_ram_bytes+size>self.stream_budget:
            path=self.root/f'stream_{hashlib.sha256(name.encode()).hexdigest()[:20]}.pt'
            if self.disk_manager is not None:self.disk_manager.ensure_space(size+65536)
            atomic_save(self.cpu_copy(value),path,torch.save)
            self.snapshot[name]=path
            self.stats['stream_spilled_bytes']+=size
        else:
            self.snapshot[name]=self.cpu_copy(value)
            if self.stream_active:self.stream_ram_bytes+=size

    def put(self, name, value):
        if self.enabled and self.mode == 'invert':
            size=value.numel()*value.element_size()
            # A source forward may be retried after cache-related GPU OOM.
            if name in self.snapshot:
                old=self.snapshot.pop(name)
                if isinstance(old,Path):old.unlink(missing_ok=True)
                else:
                    old_size=old.numel()*old.element_size()
                    if old.is_cuda:self.stream_gpu_bytes-=old_size
                    elif self.stream_active:self.stream_ram_bytes-=old_size
                self.stream_bytes-=size
                del old
            if self.gpu_room(value)>=size:
                try:self.snapshot[name]=value.detach().clone()
                except torch.cuda.OutOfMemoryError:
                    self.gpu_cache_disabled=True
                    self.gpu_cache_limit=max(0,self.stream_gpu_bytes-2**30)
                    self.stats['gpu_cache_limit_after_oom_bytes']=self.gpu_cache_limit
                    self.stats['gpu_cache_oom_fallbacks']+=1
                    self._store_cpu(name,value,size)
                else:self.stream_gpu_bytes+=size
            else:self._store_cpu(name,value,size)
            if self.stream_active:
                self.stream_bytes+=size
                self.stats['stream_peak_bytes']=max(self.stats['stream_peak_bytes'],self.stream_bytes)
                self.stats['stream_peak_ram_bytes']=max(self.stats['stream_peak_ram_bytes'],self.stream_ram_bytes)
                self.stats['stream_peak_gpu_bytes']=max(self.stats['stream_peak_gpu_bytes'],self.stream_gpu_bytes)
            self.stats['stored_tensors'] += 1

    def source_value(self,name,like):
        value=self.value(name)
        if like.is_cuda and not value.is_cuda:self.stats['h2d_bytes']+=value.numel()*value.element_size()
        return value.to(like)

    def value(self,name):
        value=self.snapshot[name]
        return torch.load(value,map_location='cpu',weights_only=True) if isinstance(value,Path) else value

    def clear_stream(self):
        for path in self.root.glob('stream_*.pt*'):
            if re.fullmatch(r'stream_[0-9a-f]{20}\.pt(?:\.tmp)?',path.name):path.unlink(missing_ok=True)
        self.snapshot={}
        self.heat_mask=None
        self.stream_active=False
        self.stream_bytes=self.stream_ram_bytes=self.stream_gpu_bytes=0
        self._mask_cache={}

    def check_budget(self,estimate,label='Attention cache'):
        limit=self.c.get('cache_max_gib',8)
        if estimate>limit*2**30:
            raise RuntimeError(f'{label} needs about {estimate/2**30:.2f} GiB, exceeding '
                f'cache_max_gib={limit}. Use attention_storage="stream" to retain heads while '
                'storing only the current denoising step, or raise cache_max_gib if enough disk is available. '
                'This limit controls disk storage, not GPU VRAM.')

    def finish(self):
        if self.stream_active:
            if self.mode=='invert' and self.step==0 and self.enabled:
                print(f'Current-step attention: {self.stream_bytes/2**30:.2f} GiB '
                      f'({self.stream_gpu_bytes/2**30:.2f} GiB on GPU; {self.stream_ram_bytes/2**30:.2f} GiB in RAM; rest on temporary disk). '
                      'Released after this edited step.',flush=True)
            if self.mode=='edit':self.clear_stream()
            return
        if self.mode == 'invert' and self.enabled:
            path = self.root/f'{self.step:03d}.pt'
            if self.disk_manager is not None:
                self.disk_manager.ensure_space(self.snapshot_bytes)
            atomic_save(self.snapshot, path, torch.save)
            size = self.snapshot_bytes = path.stat().st_size
            self.stats['cache_bytes'] += size
            if self.step == 0:
                estimate = size*self.steps
                print(f'Attention cache estimate: {estimate/2**30:.2f} GiB', flush=True)
                if estimate > self.c.get('cache_max_gib',8)*2**30:
                    path.unlink()
                    self.check_budget(estimate)
                remaining = size*(self.steps-1)
                if self.disk_manager is not None:
                    self.disk_manager.ensure_space(remaining)
                elif remaining + 2*2**30 > shutil.disk_usage(self.root).free:
                    path.unlink()
                    raise RuntimeError('Insufficient remaining attention-cache disk space')
        self.snapshot = {}

    def effective_mask(self,grid,device):
        key=(tuple(grid),str(device))
        if key not in self._mask_cache:self._mask_cache[key]=self._effective_mask(grid,device)
        return self._mask_cache[key]

    def _effective_mask(self, grid, device):
        external = None if self.mask is None else mask_grid(self.mask,grid).to(device)
        heat = None if self.heat_mask is None else mask_grid(self.heat_mask,grid).to(device)
        if self.c['mask_backend'] == 'attention': return heat
        if self.c['mask_backend'] == 'hybrid' and heat is not None and external is not None:
            return external*heat
        return external

    def blend_mask(self, grid, device):
        # Keep external segmentation at latent resolution. Attention heat uses
        # its original 32x32 working grid, matching the SD1.4 implementation.
        working = (grid[0], *self.heat_mask.shape[-2:]) if self.c['mask_backend']=='attention' and self.heat_mask is not None else grid
        weight = self.effective_mask(working, device)
        if weight is None:return None
        radius=self.c.get('blend_dilate',0)
        if radius:weight=F.max_pool2d(weight[:,None],2*radius+1,1,radius)[:,0]
        weight=mask_grid(weight[:,None],grid).to(device)
        radius=self.c.get('blend_feather',0)
        if radius:weight=F.avg_pool2d(weight[:,None],2*radius+1,1,radius,count_include_pad=False)[:,0]
        return weight.clamp(0,1)

    def image(self, name, probs, cross, frames, start=0,total=None):
        if self.mode == 'invert':
            self.put(name, probs)
            if cross and self.tokens and total is None:
                size = math.isqrt(probs.shape[-2])
                heat = probs[...,self.tokens].float().sum(-1).mean(1).reshape(frames,1,size,size)
                self.put(name+'/heat',F.interpolate(heat,size=(32,32),mode='bilinear',align_corners=False))
        elif self.mode == 'edit' and self.enabled and name in self.snapshot:
            source = self.source_value(name,probs)
            conditional = probs[-frames:]
            if cross and self.step < int(self.steps*self.c['cross_replace']):
                mapper,matched=self.alignment(probs.device)
                keep=matched.view(1,1,1,-1)
                probs[-frames:] = torch.where(keep,source[...,mapper],conditional)
                self.stats['fused_cross'] += 1
            elif not cross and self.step < int(self.steps*self.c['self_replace']):
                size = math.isqrt(total if total is not None else probs.shape[-2])
                mask = self.effective_mask((frames,size,size),probs.device)
                if mask is None:probs[-frames:] = source
                else:
                    weight=mask.reshape(frames,1,size*size,1)[:,:,start:start+probs.shape[-2]]
                    probs[-frames:] = weight*conditional + (1-weight)*source
                self.stats['fused_self'] += 1
        return probs

    def joint(self, name, probs, text_length, query_indices, selected_grid):
        """Only selected video query rows/heads. Text query rows are never fused."""
        if self.mode == 'invert':
            self.put(name,probs)
            if self.tokens and selected_grid is not None:
                tokens = [t for t in self.tokens if t < text_length]
                if tokens:
                    heat = probs[...,tokens].float().sum(-1).mean(1)[0].reshape(selected_grid)
                    heat = F.interpolate(heat[None,None],size=self.grid,mode='trilinear',align_corners=False)[0,0]
                    self.put(name+'/heat',heat[:,None])
        elif self.mode == 'edit' and self.enabled and name in self.snapshot:
            src = self.source_value(name,probs)
            target = probs[-1:]
            if self.step < int(self.steps*self.c['cross_replace']):
                mapper,matched=self.alignment(probs.device)
                keep=matched.view(1,1,1,-1)
                target[...,:text_length] = torch.where(keep,src[...,:text_length][...,mapper],target[...,:text_length])
                self.stats['fused_cross'] += 1
            if self.step < int(self.steps*self.c['self_replace']):
                mask = self.effective_mask(self.grid,probs.device)
                if mask is None:
                    target[...,text_length:] = src[...,text_length:]
                else:
                    weight = mask.flatten()[query_indices].view(1,1,-1,1)
                    target[...,text_length:] = weight*target[...,text_length:] + (1-weight)*src[...,text_length:]
                self.stats['fused_self'] += 1
            # Mixing the text and video portions can change row mass.
            target=target/target.sum(-1,keepdim=True).clamp_min(1e-8)
            probs[-1:] = target
        return probs


def bounded_sdpa(q,k,v,query_chunk=128):
    # Fused SDPA already bounds score memory internally. Chunk only when CUDA
    # cannot supply a fused backend, rather than launching dozens of tiny calls.
    if not q.is_cuda:return F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
    from torch.nn.attention import sdpa_kernel,SDPBackend
    try:
        with warnings.catch_warnings(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
            warnings.simplefilter('ignore',UserWarning)
            return F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
    except RuntimeError as error:
        if not any(text in str(error).lower() for text in ('no available kernel','no viable backend')):raise
    result=torch.empty_like(q)
    for start in range(0,q.shape[-2],query_chunk):
        result[:,:,start:start+query_chunk]=F.scaled_dot_product_attention(q[:,:,start:start+query_chunk],k,v,dropout_p=0.)
    return result


def stream_gpu_capacity(config):
    if config.get('stream_cache_device','auto')=='cpu' or not torch.cuda.is_available():return 0
    device=torch.cuda.current_device()
    free,_=torch.cuda.mem_get_info(device)
    free+=max(0,torch.cuda.memory_reserved(device)-torch.cuda.memory_allocated(device))
    room=max(0,free-int(config.get('stream_gpu_reserve_gib',4.)*2**30))
    cap=config.get('stream_cache_gpu_gib',0.)
    return min(room,int(cap*2**30)) if cap else room


def stream_ram_budget(cap_gib=0):
    """Reserve CPU working memory; overflow is spilled, never pruned."""
    import psutil
    available=psutil.virtual_memory().available
    for folder in (Path('/sys/fs/cgroup'),Path('/sys/fs/cgroup/memory')):
        limit=folder/('memory.max' if folder.name=='cgroup' else 'memory.limit_in_bytes')
        used=folder/('memory.current' if folder.name=='cgroup' else 'memory.usage_in_bytes')
        try:available=min(available,max(0,int(limit.read_text())-int(used.read_text())))
        except (OSError,ValueError):pass
    budget=max(0,available-2*2**30)
    return min(budget,int(cap_gib*2**30)) if cap_gib else budget


class JointVideoProcessor:
    """CogVideoX attention with source fusion restricted to chosen video rows/heads."""
    def __init__(self, recorder, layer):
        self.recorder,self.layer = recorder,layer

    def __call__(self,attn,hidden_states,encoder_hidden_states,attention_mask=None,image_rotary_emb=None):
        if attention_mask is not None: raise ValueError('CogVideoX attention masks are unsupported in this adapter')
        text = encoder_hidden_states.shape[1]
        hidden = torch.cat([encoder_hidden_states,hidden_states],1)
        b,n,dim = hidden.shape
        def heads(x): return x.reshape(b,n,attn.heads,-1).transpose(1,2)
        q,k,v = heads(attn.to_q(hidden)),heads(attn.to_k(hidden)),heads(attn.to_v(hidden))
        if attn.norm_q is not None:q=attn.norm_q(q)
        if attn.norm_k is not None:k=attn.norm_k(k)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            q[:,:,text:] = apply_rotary_emb(q[:,:,text:],image_rotary_emb)
            if not attn.is_cross_attention:k[:,:,text:] = apply_rotary_emb(k[:,:,text:],image_rotary_emb)
        c,r=self.recorder.c,self.recorder
        y=bounded_sdpa(q,k,v,c['attention_query_chunk'])
        record=r.wants_image(False) or r.wants_image(True)
        if record or r.wants_heat():
            f,h,w=r.grid
            stride=1
            limit=r.native_query_limit or f*h*w
            while f*math.ceil(h/stride)*math.ceil(w/stride)>limit:
                stride+=1
                if stride>max(h,w):raise ValueError('attention_max_queries must be >= latent frame count')
            indices=torch.arange(f*h*w,device=q.device).reshape(f,h,w)[:,::stride,::stride]
            shape=indices.shape
            indices=indices.flatten()
            nh=attn.heads if c['attention_heads']==0 else min(attn.heads,c['attention_heads'])
            tokens=[token for token in r.tokens if token<text]
            heat=torch.zeros(len(indices),device=q.device,dtype=torch.float32) if tokens and r.wants_heat() else None
            for start in range(0,len(indices),c['attention_query_chunk']):
                selected=indices[start:start+c['attention_query_chunk']]
                probs=(q[-1:,:nh,text+selected] @ k[-1:,:nh].transpose(-1,-2)*attn.scale).softmax(-1)
                if heat is not None:heat[start:start+len(selected)]=probs[...,tokens].float().sum(-1).mean(1)[0]
                if record:probs=r.joint(f'joint/{self.layer}/q{start:05d}',probs,text,selected,None)
                y[-1:,:nh,text+selected]=probs@v[-1:,:nh]
            if heat is not None:
                heat=F.interpolate(heat.reshape(shape)[None,None],size=r.grid,mode='trilinear',align_corners=False)[0,0]
                r.put(f'joint/{self.layer}/heat',heat[:,None])
        out=y.transpose(1,2).reshape(b,n,-1)
        out=attn.to_out[1](attn.to_out[0](out))
        return out[:,text:],out[:,:text]
