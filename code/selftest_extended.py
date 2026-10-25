"""No-download integration tests for optional backends. GPU required by default."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from options import normalized_config,validate_options
from attention_ext import SourceAttention,JointVideoProcessor,mask_grid
from backends import ImageBackend,CogVideoBackend,ImageAttentionProcessor,ModelStages
from run_extended import ExtendedEditor,to_epsilon,inverse_ddim,inversion_key
from masks import transform_mask,build_masks


def configuration(root,backend):
    c=json.loads((Path(__file__).parent/'config.example.json').read_text())
    c.update(backend=backend,model_id=None,size=16,frames=9 if backend=='cogvideox_2b' else 2,
        steps=2,model_cpu_offload=False,inversion_iterations=3,source_prompt='a cat',target_prompt='a dog',
        edit_words=[],mask_backend='file',mask_dir=str(root/'masks'),mask_dilation=0,mask_feather=0,
        mask_cache_path=str(root/'mask_cache'),mask_latent_blend=True,attention_layers=[0],attention_heads=1,
        attention_max_queries=32,output_fps=2.,mask_boxes=[],mask_points=[],mask_labels=[])
    c['attention_query_chunk']=7
    return normalized_config(c)


def tokenizer(root):
    from transformers import CLIPTokenizer
    (root/'vocab.json').write_text(json.dumps({'<|startoftext|>':0,'<|endoftext|>':1,
        'a</w>':2,'cat</w>':3,'dog</w>':4,'c':5,'a':6,'t</w>':7,'d':8,'o':9,'g</w>':10}))
    (root/'merges.txt').write_text('#version: 0.2\n')
    return CLIPTokenizer(str(root/'vocab.json'),str(root/'merges.txt'),model_max_length=77)


def scheduler_test(device):
    from diffusers import DDIMScheduler,CogVideoXDDIMScheduler
    for scheduler_class in (DDIMScheduler,CogVideoXDDIMScheduler):
        for kind in ('epsilon','v_prediction'):
            s=scheduler_class(num_train_timesteps=100,clip_sample=False,prediction_type=kind,set_alpha_to_one=False)
            s.set_timesteps(5,device=device)
            x=torch.randn(1,4,3,4,4,device=device)
            output=torch.randn_like(x)
            t=s.timesteps[1]
            epsilon=to_epsilon(output,x,t,s)
            previous=s.step(output,t,x,eta=0.).prev_sample
            recovered=inverse_ddim(epsilon,t,previous,s)
            torch.testing.assert_close(recovered,x,atol=2e-6,rtol=2e-6)
    print('PASS actual DDIM and CogVideoX scheduler round trips for epsilon and velocity')


def offload_stage_test():
    """Actual Accelerate hooks with tiny CPU models and simulated GPU capacity."""
    from types import SimpleNamespace
    from accelerate import cpu_offload_with_hook
    class Ledger:
        def __init__(self):self.resident={};self.moves=0
        def move(self,name,units,device):
            new_units=units if torch.device(device).type=='cuda' else 0
            used=sum(self.resident.values())-self.resident.get(name,0)+new_units
            if used>10:raise torch.OutOfMemoryError('simulated T5 + denoiser overlap')
            if self.resident.get(name,0)!=new_units:self.moves+=1
            self.resident[name]=new_units
    class Model(torch.nn.Module):
        def __init__(self,ledger,name,units):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(1),requires_grad=False)
            self.ledger,self.name,self.units=ledger,name,units
        def to(self,device,*args,**kwargs):
            self.ledger.move(self.name,self.units,device);return self
        def forward(self):return self.weight+1
    def pipeline(denoiser):
        ledger=Ledger();pipe=SimpleNamespace(_all_hooks=[]);previous=None
        for name,units in (('text_encoder',8),(denoiser,4),('vae',3)):
            model=Model(ledger,name,units)
            model,previous=cpu_offload_with_hook(model,execution_device='cuda:0',prev_module_hook=previous)
            setattr(pipe,name,model);pipe._all_hooks.append(previous)
        return pipe,ledger
    for denoiser in ('unet','transformer'):
        pipe,ledger=pipeline(denoiser)
        pipe.text_encoder();getattr(pipe,denoiser)()
        try:pipe.text_encoder()
        except torch.OutOfMemoryError:pass
        else:raise AssertionError('Expected original out-of-order hook overlap to exhaust simulated VRAM')
        pipe,ledger=pipeline(denoiser);stages=ModelStages(pipe,True)
        with patch('torch.cuda.empty_cache'):
            for name in ('vae','text_encoder',denoiser,'text_encoder','text_encoder',denoiser,denoiser,'vae','text_encoder',denoiser,'vae'):
                stages.activate(name)
                value=getattr(pipe,name)()
                torch.testing.assert_close(value,torch.tensor([2.]))
                assert ledger.resident[name]>0 and sum(v>0 for v in ledger.resident.values())==1
            stages.activate(denoiser);getattr(pipe,denoiser)()
            moves=ledger.moves
            stages.activate(denoiser);getattr(pipe,denoiser)()
            assert ledger.moves==moves,'Repeated denoiser forwards must not evict/reload it'
        disabled=ModelStages(SimpleNamespace(),False);disabled.activate('transformer')
    print('PASS actual Accelerate hooks: reproduced T5 overlap, exclusive stage residency, reverse transitions and denoiser retention')


def joint_processor_test(device):
    from diffusers.models.attention_processor import Attention,CogVideoXAttnProcessor2_0
    with tempfile.TemporaryDirectory() as temp:
        c=configuration(Path(temp),'cogvideox_2b')
        r=SourceAttention(temp,2,c);r.grid=(3,2,2)
        attn=Attention(query_dim=32,heads=2,dim_head=16,bias=True,qk_norm='layer_norm',eps=1e-6).to(device)
        h=torch.randn(1,12,32,device=device);text=torch.randn(1,5,32,device=device)
        reference=CogVideoXAttnProcessor2_0()(attn,h,text)
        own=JointVideoProcessor(r,0)(attn,h,text)
        for a,b in zip(reference,own):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        # RoPE path parity, tested with cos=1, sin=0 at all image positions.
        rope=(torch.ones(12,16,device=device),torch.zeros(12,16,device=device))
        reference=CogVideoXAttnProcessor2_0()(attn,h,text,image_rotary_emb=rope)
        own=JointVideoProcessor(r,0)(attn,h,text,image_rotary_emb=rope)
        for a,b in zip(reference,own):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        r.mode='invert';JointVideoProcessor(r,0)(attn,h,text)
        assert torch.cat([r.value(name) for name in sorted(r.snapshot)],dim=-2).shape==(1,1,12,17)
        r.mode='edit';r.mapper=torch.arange(5);r.matched=torch.ones(5,dtype=torch.bool)
        cfg_h=torch.randn(2,12,32,device=device);cfg_text=torch.randn(2,5,32,device=device)
        reference=CogVideoXAttnProcessor2_0()(attn,cfg_h,cfg_text)
        own=JointVideoProcessor(r,0)(attn,cfg_h,cfg_text)
        torch.testing.assert_close(own[0][0],reference[0][0],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(own[1],reference[1],atol=1e-6,rtol=1e-5)
    print('PASS native joint-attention processor equivalence, RoPE and bounded head selection')


def image_processor_test(device):
    from types import SimpleNamespace
    from diffusers.models.attention_processor import Attention
    def reference(attn,x,context=None,source=None,nh=0):
        b,n,dim=x.shape;frames=2
        q=attn.to_q(x);k=attn.to_k(x if context is None else context);v=attn.to_v(x if context is None else context)
        if context is None:
            def middle(value):return value.reshape(b//frames,frames,-1,dim)[:,:1].expand(-1,frames,-1,-1).reshape(b,-1,dim)
            k,v=middle(k),middle(v)
        def split(value):return value.reshape(b,-1,attn.heads,dim//attn.heads).transpose(1,2)
        q,k,v=split(q),split(k),split(v)
        probs=(q.float()@k.float().transpose(-1,-2)*attn.scale).softmax(-1).to(v.dtype)
        original=probs.clone()
        if source is not None:probs[-frames:,:nh]=source[:,:nh]
        out=(probs@v).transpose(1,2).reshape(b,n,dim)
        return attn.to_out[1](attn.to_out[0](out)),original
    with tempfile.TemporaryDirectory() as temp:
        for cross in (False,True):
            for cap in (1,0):
                c=configuration(Path(temp),'sd21');c.update(attention_heads=cap,attention_max_queries=32,
                    self_replace=1.,cross_replace=1.,attention_query_chunk=4)
                r=SourceAttention(temp,2,c,torch.zeros(2,1,3,3));r.tokens=[1]
                owner=SimpleNamespace(context=SimpleNamespace(frames=2),least_channel=0,recorder=r)
                processor=ImageAttentionProcessor(owner,'test')
                attn=Attention(query_dim=32,heads=4,dim_head=8,upcast_attention=True).to(device).eval()
                source=torch.randn(2,9,32,device=device);target=torch.randn(4,9,32,device=device)
                src_text=torch.randn(2,7,32,device=device) if cross else None
                dst_text=torch.randn(4,7,32,device=device) if cross else None
                expected,src_probs=reference(attn,source,src_text)
                r.begin('source',0)
                own=processor(attn,source,src_text)
                torch.testing.assert_close(own,expected,atol=1e-6,rtol=1e-5)
                nh=4 if cap==0 else cap
                assert len([name for name in r.snapshot if '/q' in name])==3
                assert all(r.value(name).shape[1]==nh for name in r.snapshot if '/q' in name)
                r.mapper=torch.arange(7);r.matched=torch.ones(7,dtype=torch.bool)
                r.begin('edit',0)
                expected,_=reference(attn,target,dst_text,src_probs,nh)
                own=processor(attn,target,dst_text)
                torch.testing.assert_close(own,expected,atol=1e-6,rtol=1e-5)
                r.finish();assert not r.snapshot
    print('PASS image chunking: full/selected heads, middle-frame self attention, cross replay and unchanged unconditional CFG')


def backend_test(root,kind,device,storage='disk',spill=False,improved=False,mask_mode='file'):
    from diffusers import (UNet2DConditionModel,AutoencoderKL,StableDiffusionPipeline,DDIMScheduler,
        CogVideoXTransformer3DModel,AutoencoderKLCogVideoX,CogVideoXPipeline,CogVideoXDDIMScheduler)
    from transformers import CLIPTextModel,CLIPTextConfig,T5EncoderModel,T5Config
    c=configuration(root,kind)
    if improved:
        from options import EDIT_PROFILES
        c.update(EDIT_PROFILES['improved'])
        c['edit_profile']='improved'
        c['attention_query_chunk']=512
    c['attention_storage']=storage
    c['mask_backend']=mask_mode
    if mask_mode in ('attention','hybrid'):c['edit_words']=['cat']
    if storage=='stream':
        c.update(attention_heads=0,cache_max_gib=1e-12,
                 attention_max_queries=0,
                 stream_cache_ram_gib=1e-12 if spill else 0.,stream_cache_device='cpu' if spill else 'auto')
        if spill:c['inversion_method']='direct'
    token=tokenizer(root)
    mask=torch.ones(c['frames'],1,16,16);mask[:,:,:,:8]=0
    recorder=SourceAttention(root/f'{kind}-{storage}-{spill}',c['steps'],c,mask)
    if kind in ('sd14','sd21'):
        model=UNet2DConditionModel(sample_size=8,in_channels=4,out_channels=4,
            down_block_types=('CrossAttnDownBlock2D','DownBlock2D'),up_block_types=('UpBlock2D','CrossAttnUpBlock2D'),
            block_out_channels=(32,64),layers_per_block=1,norm_num_groups=8,cross_attention_dim=48,attention_head_dim=4)
        vae=AutoencoderKL(block_out_channels=(32,64),down_block_types=('DownEncoderBlock2D','DownEncoderBlock2D'),
            up_block_types=('UpDecoderBlock2D','UpDecoderBlock2D'),latent_channels=4,layers_per_block=1,norm_num_groups=8,sample_size=16)
        text=CLIPTextModel(CLIPTextConfig(vocab_size=11,hidden_size=48,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,max_position_embeddings=77))
        scheduler=DDIMScheduler(num_train_timesteps=100,clip_sample=False,set_alpha_to_one=False,prediction_type='epsilon' if kind=='sd14' else 'v_prediction',steps_offset=1)
        pipe=StableDiffusionPipeline(vae,text,token,model,scheduler,safety_checker=None,feature_extractor=None,requires_safety_checker=False)
        b=ImageBackend(pipe,recorder,c,device)
        if improved:
            assert b.pipe.unet.least_channel==320
            for block in b.pipe.unet.modules():
                if hasattr(block,'attn_temporal'):
                    assert torch.count_nonzero(block.attn_temporal.to_out[0].weight)==0
                    assert torch.count_nonzero(block.attn_temporal.to_out[0].bias)==0
    else:
        model=CogVideoXTransformer3DModel(num_attention_heads=2,attention_head_dim=16,in_channels=4,out_channels=4,
            time_embed_dim=32,text_embed_dim=32,num_layers=1,sample_width=2,sample_height=2,sample_frames=9,
            patch_size=1,temporal_compression_ratio=4,max_text_seq_length=77)
        vae=AutoencoderKLCogVideoX(block_out_channels=(8,8,8,8),latent_channels=4,layers_per_block=1,
            norm_num_groups=4,sample_height=16,sample_width=16,temporal_compression_ratio=4)
        text=T5EncoderModel(T5Config(vocab_size=11,d_model=32,d_ff=64,num_layers=1,num_heads=4,d_kv=8))
        scheduler=CogVideoXDDIMScheduler(num_train_timesteps=100,clip_sample=False,set_alpha_to_one=False,prediction_type='v_prediction')
        pipe=CogVideoXPipeline(token,text,vae,model,scheduler)
        b=CogVideoBackend(pipe,recorder,c,device)
    for component in pipe.components.values():
        if isinstance(component,torch.nn.Module):component.eval().requires_grad_(False)
    b.scheduler.set_timesteps(2,device=device)
    from run_extended import source_tokens
    recorder.tokens=source_tokens(b,c['source_prompt'],c.get('edit_words',[]))
    editor=ExtendedEditor(b,recorder,c)
    images=[Image.new('RGB',(16,16),(i*10,80,100)) for i in range(c['frames'])]
    modes=[]
    forward=b.predict
    def traced(*args):
        modes.append(recorder.mode)
        return forward(*args)
    b.predict=traced
    with torch.inference_mode(),patch.object(pipe,'encode_prompt',wraps=pipe.encode_prompt) as encode_prompt:
        for prompt in (c['source_prompt'],c['target_prompt'],c.get('negative_prompt','')):b.embed(prompt)
        clean=b.encode(images)
        if storage=='disk' and kind=='sd21':
            old_limit=c['cache_max_gib'];c['cache_max_gib']=1e-12
            try:b.prepare_attention_cache(clean)
            except RuntimeError as error:assert 'attention_storage="stream"' in str(error)
            else:raise AssertionError('Expected budget rejection before inversion files exist')
            assert not list(recorder.root.glob('*.pt'))
            c['cache_max_gib']=old_limit
            planned=b.prepare_attention_cache(clean)
        with patch.object(recorder,'finish',wraps=recorder.finish) as finish:
            noisy=editor.invert(clean)
        iterations=1 if c['inversion_method']=='direct' else 3
        assert modes==(['off']*(2*iterations) if storage=='stream' else ['off','off','invert']*2) and finish.call_count==2
        assert len(list(recorder.root.glob('[0-9][0-9][0-9].pt')))==(0 if storage=='stream' else 2)
        assert len(list(recorder.root.glob('latent_*.pt')))==(0 if storage=='stream' else 2)
        if storage=='stream':
            if kind in ('sd14','sd21'):
                # An actual capacity shortage must lower spatial resolution,
                # retaining every head; sufficient space restores the full grid.
                c['attention_max_queries']=16
                small=b.prepare_attention_cache(noisy)
                c['attention_max_queries']=0
                with patch('backends.stream_ram_budget',return_value=0), patch('backends.stream_gpu_capacity',return_value=0), \
                     patch('shutil.disk_usage',return_value=type('Disk',(),{'free':small+int(c['cache_reserve_gib']*2**30)})()):
                    b.prepare_attention_cache(noisy)
                assert recorder.image_query_limit==16 and c['attention_heads']==0
                b.prepare_attention_cache(noisy)
                assert recorder.image_query_limit==64
            else:
                c['attention_max_queries']=3;small=b.prepare_attention_cache(noisy)
                c['attention_max_queries']=0
                with patch('backends.stream_ram_budget',return_value=0), patch('backends.stream_gpu_capacity',return_value=0), \
                     patch('shutil.disk_usage',return_value=type('Disk',(),{'free':small+int(c['cache_reserve_gib']*2**30)})()):
                    b.prepare_attention_cache(noisy)
                assert recorder.native_query_limit==3 and c['attention_heads']==0
                b.prepare_attention_cache(noisy)
                assert recorder.native_query_limit==12
            # Independent offline oracle: retain every source denoising step,
            # then replay those maps in a separate edit pass. Pairing must give
            # the same result while keeping no past-step maps.
            source=noisy.clone();embeddings=b.embed(c['source_prompt'])
            for i,t in enumerate(b.scheduler.timesteps):
                recorder.begin('source',i)
                prediction=b.predict(b.scheduler.scale_model_input(source,t),t,embeddings)
                source=b.scheduler.step(prediction,t,source,eta=0.).prev_sample.to(noisy.dtype)
                snapshot={name:recorder.value(name).clone() for name in recorder.snapshot}
                if i<int(c['steps']*max(c['self_replace'],c['cross_replace'])) or mask_mode in ('attention','hybrid'):assert snapshot
                for name,value in snapshot.items():
                    if name.endswith('/heat'):continue
                    if kind in ('sd14','sd21'):
                        module=b.pipe.unet.backbone.get_submodule(name.rsplit('/q',1)[0])
                        assert value.shape[1]==module.heads
                    else:assert value.shape[1]==b.pipe.transformer.transformer_blocks[0].attn1.heads
                torch.save(snapshot,recorder.root/f'{c["steps"]-i-1:03d}.pt')
                torch.save(source,recorder.root/f'latent_{c["steps"]-i-1:03d}.pt')
                recorder.clear_stream()
            reference=editor.denoise(noisy,c['target_prompt'],True)
            for file in recorder.root.glob('*.pt'):file.unlink()
            result,paired_source=editor.denoise_stream(noisy,c['target_prompt'])
            torch.testing.assert_close(paired_source,source,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(result,reference,atol=1e-6,rtol=1e-5)
            assert not recorder.snapshot and not recorder.stream_active
            assert not list(recorder.root.glob('*.pt'))
            assert recorder.stats['stream_peak_bytes']>0
            if spill:assert recorder.stats['stream_spilled_bytes']>0
        else:result=editor.denoise(noisy,c['target_prompt'],True)
        if storage=='disk' and kind=='sd21':
            assert sum(p.stat().st_size for p in recorder.root.glob('[0-9][0-9][0-9].pt'))<=planned
        assert torch.isfinite(result).all() and result.device==clean.device
        # Exact latent preservation outside externally supplied mask at last step.
        weight=recorder.blend_mask(tuple(result.shape[2:]),device) if mask_mode=='file' else None
        if mask_mode=='file':torch.testing.assert_close(result[...,weight==0],(source if storage=='stream' else clean)[...,weight==0])
        else:assert editor.consensus_mask() is not None
        frames=b.decode(result)
        assert len(frames)==c['frames'] and frames[0].size==(16,16)
        assert b.calls==(2*iterations+8 if storage=='stream' else 8)
        assert recorder.stats['fused_self']>0
        if storage=='stream':
            predict=b.predict
            def fail_edit(*args):
                if recorder.mode=='edit':raise RuntimeError('simulated edit failure')
                return predict(*args)
            with patch.object(b,'predict',side_effect=fail_edit):
                try:editor.denoise_stream(noisy,c['target_prompt'])
                except RuntimeError as error:assert 'simulated edit failure' in str(error)
                else:raise AssertionError('Expected injected edit failure')
            assert not recorder.snapshot and not recorder.stream_active and recorder.mode=='off'
            assert not list(recorder.root.glob('*.pt'))
        assert [call.args[0] for call in encode_prompt.call_args_list]==[c['source_prompt'],c['target_prompt'],c.get('negative_prompt','')]
        first=b.embed(c['source_prompt']);expected=first.clone();first.fill_(0)
        torch.testing.assert_close(b.embed(c['source_prompt']),expected)
        assert encode_prompt.call_count==3
        assert all(value.device.type=='cpu' and not value.requires_grad for value in b._prompt_cache.values())
    print('PASS',kind,storage,'spill' if spill else 'RAM','improved' if improved else 'original',mask_mode,'tiny encode/invert/edit/decode, attention replay and background preservation')


def mask_tests(root):
    from run import prepare_frames
    frames=root/'input';frames.mkdir();files=root/'masks';files.mkdir();out=root/'out';out.mkdir()
    for i in range(4):
        Image.new('RGB',(40,24),(70,20,30)).save(frames/f'{i:05d}.png')
        a=np.zeros((24,40),dtype='uint8');a[:,10:30]=255;Image.fromarray(a).save(files/f'{i:05d}.png')
    c=configuration(root,'sd21');c.update(video_url='',video_path=str(frames),input_fps=4,
        start_frame=0,frame_stride=1,clip_seconds=1.,frames=2,size=16)
    paths,fps,indices=prepare_frames(c,out/'source');c['output_fps']=fps
    images=[Image.open(p) for p in paths]
    masks,key=build_masks(c,indices,images,out,device='cpu')
    assert masks.shape==(2,1,16,16) and masks.max()==1 and masks.min()==0
    assert (out/'mask_preview.mp4').is_file()
    assert torch.equal(mask_grid(torch.arange(9.)[:,None,None,None],(3,1,1))[:,0,0],torch.tensor([0.,4.,8.]))
    changed=dict(c,target_prompt='other edit')
    _,key2=build_masks(changed,indices,images,out,device='cpu');assert key==key2
    print('PASS mask coordinates, temporal pooling, file-mask cache reuse and preview exports')
    # Exercise our SAM tracking orchestration with a contract-compatible predictor;
    # this does not load pretrained segmentation weights or assess mask accuracy.
    import builtins
    import contextlib
    from masks import sam_track
    class Predictor:
        def __init__(self):self.prompts=[]
        def init_state(self,video_path,offload_video_to_cpu,offload_state_to_cpu):
            assert len(list(Path(video_path).glob('*.jpg')))==3
            return {}
        def add_new_points_or_box(self,inference_state,frame_idx,obj_id,points=None,labels=None,box=None):
            self.prompts.append((frame_idx,obj_id))
        def propagate_in_video(self,state):
            for i in range(3):
                logits=torch.full((1,1,24,40),-1.)
                logits[:,:,:,10:30]=1.
                yield i,[1],logits
        def reset_state(self,state):pass
    predictor=Predictor();dest=root/'sam-result';dest.mkdir()
    c.update(mask_backend='grounded_sam2',resolved_sam_revision='test-revision',
             mask_corrections=[dict(frame_index=1,object_id=1,points=[[20,10]],labels=[1])])
    real_import=builtins.__import__
    sam_imports=[]
    def without_sam(name,*args,**kwargs):
        if name=='sam2' or name.startswith('sam2.'):
            sam_imports.append(name)
            raise ModuleNotFoundError("No module named 'sam2'",name='sam2')
        return real_import(name,*args,**kwargs)
    with patch('builtins.__import__',side_effect=without_sam), \
         patch('huggingface_hub.hf_hub_download',return_value='/tmp/test-checkpoint'), \
         patch('masks.grounding_boxes',return_value=np.array([[10,0,30,24]],dtype='float32')), \
         patch('torch.autocast',return_value=contextlib.nullcontext()):
        assert build_masks(dict(c,mask_backend='attention'),[],[],out,device='cpu')==(None,None)
        sam_track(c,[0,1,2],[0,2],dest,'cpu',predictor_factory=lambda *args,**kwargs:predictor)
        assert not sam_imports,'Attention masks and mocked tracking must not import SAM 2'
        try:sam_track(c,[0,1,2],[0,2],dest,'cpu')
        except RuntimeError as error:
            assert 'requirements-segmentation.txt' in str(error)
            assert isinstance(error.__cause__,ModuleNotFoundError)
        else:raise AssertionError('Real SAM tracking must report its missing optional dependency')
    assert predictor.prompts==[(0,1),(1,1)]
    assert len(list(dest.glob('*.png')))==2
    assert np.asarray(Image.open(dest/'00000.png'))[:,10:30].min()==255
    print('PASS SAM tracking orchestration without SAM installed: dense input, prompts, corrections, selected masks; missing-dependency guidance')


def cache_key_test(root):
    path=root/'fingerprint-input';path.write_bytes(b'source')
    c=configuration(root,'sd21');c['attention_storage']='stream'
    key=inversion_key(c,[path],'checkpoint-sha')
    assert key==inversion_key(dict(c,target_prompt='different',mask_backend='attention',attention_heads=0,
        attention_max_queries=0,attention_query_chunk=512,stream_cache_device='cpu',stream_profile=True,edit_words=['cat']),[path],'checkpoint-sha')
    assert key!=inversion_key(dict(c,inversion_iterations=4),[path],'checkpoint-sha')
    assert key!=inversion_key(c,[path],'new-checkpoint-sha')
    c['attention_storage']='disk'
    assert inversion_key(c,[path],'checkpoint-sha')!=inversion_key(dict(c,attention_heads=0),[path],'checkpoint-sha')
    assert inversion_key(c,[path],'checkpoint-sha')!=inversion_key(dict(c,attention_query_chunk=512),[path],'checkpoint-sha')
    print('PASS streaming noisy-latent reuse across target/mask/attention-region changes; solver/checkpoint invalidation')


def main(device):
    torch.set_num_threads(2);torch.manual_seed(0)
    scheduler_test(device);offload_stage_test();joint_processor_test(device);image_processor_test(device)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        cache_key_test(root)
        mask_tests(root)
        for kind in ('sd14','sd21','cogvideox_2b'):backend_test(root,kind,device)
        for kind in ('sd14','sd21','cogvideox_2b'):
            backend_test(root,kind,device,'stream')
            backend_test(root,kind,device,'stream',spill=True)
            for mask_mode in ('file','attention','hybrid'):
                backend_test(root,kind,device,'stream',improved=True,mask_mode=mask_mode)
    print('All optional-backend wiring checks passed. Pretrained visual quality and Kaggle VRAM are not established by tiny tests.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--development-cpu',action='store_true');args=parser.parse_args()
    if not args.development_cpu and not torch.cuda.is_available():raise RuntimeError('CUDA required for the notebook smoke test; no CPU fallback')
    main(torch.device('cpu' if args.development_cpu else 'cuda:0'))
