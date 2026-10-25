"""Image-model adapters and a native CogVideoX adapter; pretrained layers come from Diffusers."""
from types import MethodType
import math
import torch
from torch import nn
import torch.nn.functional as F
from engine import TemporalConv, VideoGroupNorm, VideoContext
from attention_ext import bounded_sdpa, JointVideoProcessor,stream_ram_budget,stream_gpu_capacity


def stream_storage_capacity(recorder,config,desired):
    """Reclaim eligible old caches before reducing the recorded query grid."""
    import shutil
    ram=stream_ram_budget(config.get('stream_cache_ram_gib',0))+stream_gpu_capacity(config)
    if recorder.disk_manager is not None:
        try:recorder.disk_manager.ensure_space(max(0,desired-ram))
        except RuntimeError as error:
            if not str(error).startswith('Insufficient attention-cache disk:'):raise
    return ram+max(0,shutil.disk_usage(recorder.root).free-int(config.get('cache_reserve_gib',2)*2**30))


class ModelStages:
    """Make non-linear pipeline stages safe with Accelerate's ordered hooks."""
    def __init__(self,pipe,enabled):
        self.pipe,self.enabled,self.active=pipe,enabled,None

    def activate(self,name):
        if not self.enabled or self.active==name:return
        target=getattr(self.pipe,name)
        # The default hook evicts only its predecessor. Inversion revisits T5
        # after the transformer, and VAE encoding also precedes text encoding.
        for hook in self.pipe._all_hooks:
            if hook.model is not target:hook.offload()
        torch.cuda.empty_cache()
        self.active=name


class ImageAttentionProcessor:
    def __init__(self,owner,name):self.owner,self.name=owner,name

    def __call__(self,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,temb=None,*args,**kwargs):
        if hidden_states.ndim!=3 or attention_mask is not None:
            raise ValueError('The image-video adapter expects unmasked token attention')
        residual=hidden_states
        cross=encoder_hidden_states is not None
        if attn.spatial_norm is not None:hidden_states=attn.spatial_norm(hidden_states,temb)
        if attn.group_norm is not None:hidden_states=attn.group_norm(hidden_states.transpose(1,2)).transpose(1,2)
        context=hidden_states if not cross else encoder_hidden_states
        if cross and attn.norm_cross:context=attn.norm_encoder_hidden_states(context)
        q,k,v=attn.to_q(hidden_states),attn.to_k(context),attn.to_v(context)
        b,n,dim=q.shape
        f=self.owner.context.frames
        if not cross and dim>=self.owner.least_channel:
            def middle(x):
                x=x.reshape(b//f,f,-1,dim)
                return x[:,(f-1)//2:(f-1)//2+1].expand(-1,f,-1,-1).reshape(b,-1,dim)
            k,v=middle(k),middle(v)
        def split(x):return x.reshape(b,-1,attn.heads,dim//attn.heads).transpose(1,2)
        q,k,v=split(q),split(k),split(v)
        if attn.norm_q is not None:q=attn.norm_q(q)
        if attn.norm_k is not None:k=attn.norm_k(k)
        r=self.owner.recorder
        record=r.wants_image(cross)
        need_heat=cross and r.wants_heat()
        if (record or need_heat) and (r.image_query_limit==0 or n<=r.image_query_limit):
            # Only conditional frames and configured heads need explicit maps.
            # Unselected heads retain the model's own attention, including CFG.
            nh=attn.heads if r.c['attention_heads']==0 else min(attn.heads,r.c['attention_heads'])
            y=torch.empty_like(q)
            if b>f:y[:-f]=bounded_sdpa(q[:-f],k[:-f],v[:-f],r.c['attention_query_chunk'])
            if nh<attn.heads:
                y[-f:,nh:]=bounded_sdpa(q[-f:,nh:],k[-f:,nh:],v[-f:,nh:],r.c['attention_query_chunk'])
            selected_q,selected_k=q[-f:,:nh],k[-f:,:nh]
            # SD2.x asks for attention upcasting in its checkpoint configuration.
            if attn.upcast_attention:selected_q,selected_k=selected_q.float(),selected_k.float()
            heat=torch.zeros(f,n,device=q.device,dtype=torch.float32) if need_heat else None
            for start in range(0,n,r.c['attention_query_chunk']):
                stop=min(n,start+r.c['attention_query_chunk'])
                scores=selected_q[:,:,start:stop]@selected_k.transpose(-1,-2)
                probs=(scores*attn.scale).softmax(-1).to(v.dtype)
                if heat is not None:heat[:,start:stop]=probs[...,r.tokens].float().sum(-1).mean(1)
                if record:probs=r.image(f'{self.name}/q{start:05d}',probs,cross,f,start,n)
                y[-f:,:nh,start:stop]=probs@v[-f:,:nh]
            if heat is not None:
                side=int(n**.5)
                r.put(self.name+'/heat',F.interpolate(heat.reshape(f,1,side,side),size=(32,32),mode='bilinear',align_corners=False))
        else:y=bounded_sdpa(q,k,v,r.c['attention_query_chunk'])
        y=y.transpose(1,2).reshape(b,n,dim)
        y=attn.to_out[1](attn.to_out[0](y))
        if attn.residual_connection:y=y+residual
        return y/attn.rescale_output_factor


class ModernVideoUNet(nn.Module):
    """The original temporal adaptation using current Diffusers attention APIs."""
    def __init__(self,backbone,recorder,rank=160,least_channel=640,zero_temporal_bias=False):
        super().__init__()
        from diffusers.models.attention_processor import Attention
        self.backbone,self.recorder,self.least_channel=backbone,recorder,least_channel
        self.context=VideoContext()
        for name,module in list(backbone.named_modules()):
            parent_name,_,child=name.rpartition('.')
            parent=backbone.get_submodule(parent_name) if parent_name else backbone
            if isinstance(module,nn.Conv2d) and module.kernel_size[0]>1:
                setattr(parent,child,TemporalConv(module,self.context,rank))
            elif isinstance(module,nn.GroupNorm) and ('.resnets.' in name or name=='conv_norm_out'):
                setattr(parent,child,VideoGroupNorm(module,self.context))
        for name,module in list(backbone.named_modules()):
            if module.__class__.__name__=='BasicTransformerBlock':
                dim=module.norm1.normalized_shape[0]
                module.norm_temporal=nn.LayerNorm(dim)
                module.attn_temporal=Attention(query_dim=dim,heads=module.attn1.heads,dim_head=dim//module.attn1.heads,bias=False)
                nn.init.zeros_(module.attn_temporal.to_out[0].weight)
                if zero_temporal_bias:nn.init.zeros_(module.attn_temporal.to_out[0].bias)
                owner=self
                original=module.forward
                def temporal(block,*args,_forward=original,**kwargs):
                    y=_forward(*args,**kwargs)
                    b,n,d=y.shape;f=owner.context.frames
                    z=y.reshape(b//f,f,n,d).transpose(1,2).reshape(-1,f,d)
                    z=z+block.attn_temporal(block.norm_temporal(z))
                    return z.reshape(b//f,n,f,d).transpose(1,2).reshape_as(y)
                module.forward=MethodType(temporal,module)
        for name,module in backbone.named_modules():
            if isinstance(module,Attention) and not name.endswith('attn_temporal'):
                module.set_processor(ImageAttentionProcessor(self,name))

    @property
    def dtype(self):return next(self.parameters()).dtype

    @property
    def device(self):return next(self.parameters()).device

    @property
    def config(self):return self.backbone.config

    def forward(self,x,t,encoder_hidden_states,**kwargs):
        # The wrapper keeps the flattened batch compatible with pipeline offload hooks.
        return self.backbone(x,t,encoder_hidden_states=encoder_hidden_states,**kwargs)


class ImageBackend:
    def __init__(self,pipe,recorder,config,device='cuda:0'):
        self.pipe,self.recorder,self.c,self.device=pipe,recorder,config,torch.device(device)
        self.tokenizer,self.scheduler=pipe.tokenizer,pipe.scheduler
        self.dtype=next(pipe.unet.parameters()).dtype
        if config.get('vae_fp32',False):pipe.vae.to(dtype=torch.float32)
        pipe.unet=ModernVideoUNet(pipe.unet,recorder,least_channel=config.get('sparse_least_channel',640),
            zero_temporal_bias=config.get('zero_temporal_bias',False)).to(dtype=self.dtype)
        self.calls=0
        self.scheduler.config  # materialize before hooks
        if config.get('model_cpu_offload',True) and self.device.type=='cuda':pipe.enable_model_cpu_offload(device=self.device)
        else:pipe.to(self.device)
        self.stages=ModelStages(pipe,config.get('model_cpu_offload',True) and self.device.type=='cuda')
        self._prompt_cache={}
        pipe.enable_vae_slicing()

    def ids(self,prompt):return self.tokenizer(prompt,padding='max_length',max_length=77,truncation=True).input_ids

    def embed(self,prompt):
        if len(self.tokenizer.encode(prompt))>77:raise ValueError('Prompt exceeds the CLIP 77-token limit')
        if prompt not in self._prompt_cache:
            self.stages.activate('text_encoder')
            value=self.pipe.encode_prompt(prompt,self.device,1,False)[0]
            self._prompt_cache[prompt]=value.detach().to('cpu',copy=True)
        return self._prompt_cache[prompt].to(self.device,copy=True)

    def encode(self,images):
        import numpy as np
        self.stages.activate('vae')
        x=torch.from_numpy(np.stack([np.asarray(im) for im in images]).copy()).permute(0,3,1,2).to(self.device,next(self.pipe.vae.parameters()).dtype)/127.5-1
        scale=self.pipe.vae.config.scaling_factor
        z=torch.cat([self.pipe.vae.encode(frame).latent_dist.mode() for frame in x.split(1)])*scale
        return z.permute(1,0,2,3).unsqueeze(0).to(self.dtype)

    def predict(self,z,t,embeddings):
        self.stages.activate('unet')
        z=z.to(device=self.device,dtype=self.dtype)
        embeddings=embeddings.to(device=self.device,dtype=self.dtype)
        b,c,f,h,w=z.shape
        self.pipe.unet.context.frames=f
        self.calls+=1
        x=z.permute(0,2,1,3,4).reshape(b*f,c,h,w)
        y=self.pipe.unet(x,t,encoder_hidden_states=embeddings.repeat_interleave(f,0)).sample
        return y.reshape(b,f,c,h,w).permute(0,2,1,3,4)

    def prepare_attention_cache(self,clean):
        """Estimate stored tensors from the loaded UNet and actual latent shape."""
        r=self.recorder
        if not r.enabled:return
        self.stages.activate('unet')
        _,_,frames,h,w=clean.shape
        grids=[]
        for block in self.pipe.unet.backbone.down_blocks:
            grids.append((h,w))
            if block.downsamplers:
                h,w=(h+1)//2,(w+1)//2
        middle=(h,w)
        entries=[]
        for name,module in self.pipe.unet.backbone.named_modules():
            if not isinstance(getattr(module,'processor',None),ImageAttentionProcessor):continue
            parts=name.split('.')
            if parts[0]=='down_blocks':height,width=grids[int(parts[1])]
            elif parts[0]=='up_blocks':height,width=grids[-1-int(parts[1])]
            elif parts[0]=='mid_block':height,width=middle
            else:raise ValueError(f'Cannot estimate image attention grid for {name}')
            queries=height*width
            cross=parts[-1]=='attn2'
            keys=77 if cross else queries
            heads=module.heads if self.c['attention_heads']==0 else min(module.heads,self.c['attention_heads'])
            size=frames*heads*queries*keys*clean.element_size()
            tensors=(queries+self.c['attention_query_chunk']-1)//self.c['attention_query_chunk']
            if cross and r.tokens:
                size+=frames*32*32*4;tensors+=1
            entries.append((queries,size+4096*tensors))
        def stored_size(limit):return 65536+sum(size for queries,size in entries if queries<=limit)
        limit=self.c['attention_max_queries']
        if limit==0:
            choices=sorted({queries for queries,_ in entries},reverse=True)
            capacity=stream_storage_capacity(r,self.c,stored_size(max(choices)))
            limit=next((candidate for candidate in choices if stored_size(candidate)<=capacity),None)
            if limit is None:
                raise RuntimeError('Even one full-head attention resolution exceeds available RAM + spill disk. '
                    'Free disk, reduce frames/resolution, or explicitly select fewer attention_heads.')
            if limit<max(choices):
                print(f'Attention storage requires a spatial limit: {limit} queries per image layer; '
                      'all heads retained. Higher resolutions exceed available RAM + spill disk.',flush=True)
        r.image_query_limit=limit
        r.stats['effective_image_max_queries']=limit
        # Conservative allowance for torch.save metadata, alignment and headers.
        per_step=stored_size(limit)
        if self.c.get('attention_storage')=='stream':
            heads='all' if self.c['attention_heads']==0 else self.c['attention_heads']
            print(f'Streaming source attention: {heads} heads, <= {limit} queries; ~{per_step/2**30:.2f} GiB '
                  'per step (GPU cache with RAM/disk overflow), released after each edited step. '
                  'No inversion attention archive.',flush=True)
            return per_step
        estimate=per_step*r.steps
        r.check_budget(estimate,'Image attention cache preflight')
        heads='all' if self.c['attention_heads']==0 else self.c['attention_heads']
        print(f'Image attention recording: {heads} heads per layer, '
              f'<= {self.c["attention_max_queries"]} spatial queries; estimated disk {estimate/2**30:.2f} GiB',flush=True)
        if r.disk_manager is not None:r.disk_manager.ensure_space(estimate)
        return estimate

    def decode(self,z):
        from PIL import Image
        self.stages.activate('vae')
        frames=[]
        for frame in z[0].permute(1,0,2,3).split(1):
            x=(self.pipe.vae.decode((frame/self.pipe.vae.config.scaling_factor).to(next(self.pipe.vae.parameters()).dtype)).sample/2+.5).clamp(0,1)
            if not torch.isfinite(x).all():raise RuntimeError('VAE decoding produced nonfinite pixels; enable vae_fp32')
            frames.append(Image.fromarray((x[0].permute(1,2,0).float().cpu().numpy()*255).round().astype('uint8')))
        return frames


class CogVideoBackend:
    def __init__(self,pipe,recorder,config,device='cuda:0'):
        self.pipe,self.recorder,self.c,self.device=pipe,recorder,config,torch.device(device)
        self.tokenizer,self.scheduler=pipe.tokenizer,pipe.scheduler
        self.dtype=next(pipe.transformer.parameters()).dtype
        if config.get('vae_fp32',True):pipe.vae.to(dtype=torch.float32)
        self.calls=0
        count=len(pipe.transformer.transformer_blocks)
        if any(i>=count for i in config['attention_layers']):raise ValueError(f'attention_layers must be <{count}')
        for i in config['attention_layers']:
            pipe.transformer.transformer_blocks[i].attn1.set_processor(JointVideoProcessor(recorder,i))
        if config.get('model_cpu_offload',True) and self.device.type=='cuda':pipe.enable_model_cpu_offload(device=self.device)
        else:pipe.to(self.device)
        self.stages=ModelStages(pipe,config.get('model_cpu_offload',True) and self.device.type=='cuda')
        self._prompt_cache={}
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling(tile_sample_min_height=max(64,pipe.vae.tile_sample_min_height),
                               tile_sample_min_width=max(64,pipe.vae.tile_sample_min_width))
        self.max_tokens=pipe.transformer.config.max_text_seq_length
        self.rope=None

    def ids(self,prompt):
        return self.tokenizer(prompt,padding='max_length',max_length=self.max_tokens,truncation=True,add_special_tokens=True).input_ids

    def embed(self,prompt):
        if len(self.tokenizer.encode(prompt))>self.max_tokens:raise ValueError('Prompt exceeds native model text token limit')
        if prompt not in self._prompt_cache:
            self.stages.activate('text_encoder')
            value=self.pipe.encode_prompt(prompt,device=self.device,num_videos_per_prompt=1,do_classifier_free_guidance=False,max_sequence_length=self.max_tokens)[0]
            self._prompt_cache[prompt]=value.detach().to('cpu',copy=True)
        return self._prompt_cache[prompt].to(self.device,copy=True)

    def encode(self,images):
        import numpy as np
        self.stages.activate('vae')
        x=torch.from_numpy(np.stack([np.asarray(im) for im in images]).copy()).permute(3,0,1,2)[None].to(self.device,next(self.pipe.vae.parameters()).dtype)/127.5-1
        return (self.pipe.vae.encode(x).latent_dist.mode()*self.pipe.vae.config.scaling_factor).to(self.dtype)

    def predict(self,z,t,embeddings):
        self.stages.activate('transformer')
        z=z.to(device=self.device,dtype=self.dtype)
        embeddings=embeddings.to(device=self.device,dtype=self.dtype)
        f,h,w=z.shape[2:]
        p=self.pipe.transformer.config.patch_size
        self.recorder.grid=(f,h//p,w//p)
        if self.pipe.transformer.config.use_rotary_positional_embeddings and self.rope is None:
            self.rope=self.pipe._prepare_rotary_positional_embeddings(self.c['size'],self.c['size'],f,self.device)
        self.calls+=1
        ts=t.expand(z.shape[0]) if torch.is_tensor(t) else torch.full((z.shape[0],),t,device=self.device)
        y=self.pipe.transformer(hidden_states=z.permute(0,2,1,3,4),encoder_hidden_states=embeddings,
            timestep=ts,image_rotary_emb=self.rope,return_dict=False)[0]
        return y.permute(0,2,1,3,4)

    def prepare_attention_cache(self,clean):
        r=self.recorder
        if not r.enabled:return
        self.stages.activate('transformer')
        f,h,w=clean.shape[2:];patch=self.pipe.transformer.config.patch_size
        h,w=h//patch,w//patch
        layers=[self.pipe.transformer.transformer_blocks[i].attn1 for i in set(self.c['attention_layers'])]
        heads=[module.heads if self.c['attention_heads']==0 else min(module.heads,self.c['attention_heads']) for module in layers]
        keys=f*h*w+self.max_tokens
        def stored_size(queries):
            tensors=len(layers)*math.ceil(queries/self.c['attention_query_chunk'])
            heat=len(layers)*f*h*w*4 if r.tokens else 0
            return queries*keys*sum(heads)*clean.element_size()+heat+4096*tensors+65536
        limit=self.c['attention_max_queries']
        if limit==0:
            capacity=stream_storage_capacity(r,self.c,stored_size(f*h*w))
            for stride in range(1,max(h,w)+1):
                queries=f*math.ceil(h/stride)*math.ceil(w/stride)
                if stored_size(queries)<=capacity:
                    limit=queries;break
            if limit==0:
                raise RuntimeError('Even one source query per latent frame exceeds available RAM + spill disk. '
                    'Free disk or explicitly reduce selected layers, frames, resolution or attention_heads.')
            if limit<f*h*w:
                print(f'Attention storage requires a native query limit: {limit}; '
                      'all configured heads and layers retained.',flush=True)
        if limit<f:raise ValueError('attention_max_queries must be >= the native latent frame count')
        r.native_query_limit=limit
        r.stats['effective_native_max_queries']=limit
        stride=1
        while f*math.ceil(h/stride)*math.ceil(w/stride)>limit:stride+=1
        per_step=stored_size(f*math.ceil(h/stride)*math.ceil(w/stride))
        if self.c.get('attention_storage')=='stream':
            print(f'Streaming native attention: up to {max(heads)} heads in {len(layers)} layers, '
                  f'<= {limit} video queries; ~{per_step/2**30:.2f} GiB per step (GPU cache with RAM/disk overflow).',flush=True)
            return per_step
        estimate=per_step*r.steps
        r.check_budget(estimate,'Native attention cache preflight')
        if r.disk_manager is not None:r.disk_manager.ensure_space(estimate)
        return estimate

    def decode(self,z):
        from PIL import Image
        self.stages.activate('vae')
        x=(self.pipe.vae.decode((z/self.pipe.vae.config.scaling_factor).to(next(self.pipe.vae.parameters()).dtype)).sample/2+.5).clamp(0,1)
        if not torch.isfinite(x).all():raise RuntimeError('VAE decoding produced nonfinite pixels; enable vae_fp32')
        arrays=(x[0].permute(1,2,3,0).float().cpu().numpy()*255).round().astype('uint8')
        return [Image.fromarray(a) for a in arrays]


def resolve_checkpoint(c):
    """Resolve mutable Hub refs once so inference and cache identity use the same revision."""
    from pathlib import Path
    import hashlib,json,os
    if c.get('model_path'):
        path=Path(c['model_path']).resolve()
        if not path.is_dir():raise FileNotFoundError(path)
        signature=[]
        for p in sorted(path.rglob('*')):
            if p.is_file():signature.append((str(p.relative_to(path)),p.stat().st_size,p.stat().st_mtime_ns))
        return str(path),None,hashlib.sha256(json.dumps(signature).encode()).hexdigest()
    from huggingface_hub import model_info
    revision=model_info(c['model_id'],revision=c['model_revision'],token=os.environ.get('HF_TOKEN')).sha
    return c['model_id'],revision,revision


def load_backend(c,recorder,source,revision):
    if c['backend']=='cogvideox_2b':
        from diffusers import CogVideoXPipeline,CogVideoXDDIMScheduler
        pipe=CogVideoXPipeline.from_pretrained(source,revision=revision,torch_dtype=torch.float16)
        pipe.scheduler=CogVideoXDDIMScheduler.from_config(pipe.scheduler.config,clip_sample=False,timestep_spacing='leading')
        result=CogVideoBackend(pipe,recorder,c)
    else:
        from diffusers import StableDiffusionPipeline,DDIMScheduler
        pipe=StableDiffusionPipeline.from_pretrained(source,revision=revision,torch_dtype=torch.float16,
            safety_checker=None,requires_safety_checker=False)
        pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config,clip_sample=False,timestep_spacing='leading')
        result=ImageBackend(pipe,recorder,c)
    if pipe.scheduler.config.prediction_type not in ('epsilon','v_prediction'):
        raise ValueError('Only epsilon and velocity prediction checkpoints are supported')
    pipe.set_progress_bar_config(disable=True)
    for component in pipe.components.values():
        if isinstance(component,nn.Module):component.eval().requires_grad_(False)
    return result
