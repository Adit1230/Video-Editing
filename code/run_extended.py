"""Configurable SD1.4/SD2.1 and native CogVideoX inversion editing."""
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from inversion import fixed_point_solve
from options import normalized_config,validate_options
from run import prepare_frames,export_video
from attention_ext import SourceAttention,mask_grid
from masks import build_masks
from backends import resolve_checkpoint,load_backend
from cache_manager import CacheManager,atomic_save,cleanup_incomplete


def to_epsilon(output,guess,t,scheduler):
    alpha=scheduler.alphas_cumprod[int(t)].to(guess)
    kind=scheduler.config.prediction_type
    if kind=='epsilon':return output
    if kind=='v_prediction':return alpha.sqrt()*output+(1-alpha).sqrt()*guess
    raise ValueError(f'Unsupported prediction_type: {kind}')


def inverse_ddim(epsilon,t,previous,scheduler):
    index=int(t)-scheduler.config.num_train_timesteps//scheduler.num_inference_steps
    alpha_previous=scheduler.alphas_cumprod[index] if index>=0 else scheduler.final_alpha_cumprod
    a=alpha_previous.to(previous)
    b=scheduler.alphas_cumprod[int(t)].to(previous)
    return (b/a).sqrt()*previous+((1-b).sqrt()-(b/a).sqrt()*(1-a).sqrt())*epsilon


def inversion_key(c,paths,checkpoint_identity):
    keys=('backend','model_id','model_revision','source_prompt','steps','size','frames','seed',
          'inversion_method','inversion_iterations','inversion_relaxation','cache_attention',
          'sparse_least_channel','zero_temporal_bias','attention_storage','attention_layers','attention_heads','attention_max_queries','attention_query_chunk','edit_words','vae_fp32')
    if c.get('attention_storage')=='stream':
        keys=tuple(k for k in keys if k not in ('cache_attention','attention_layers','attention_heads','attention_max_queries','attention_query_chunk','edit_words'))
    h=hashlib.sha256(json.dumps({k:c.get(k) for k in keys},sort_keys=True).encode())
    h.update(checkpoint_identity.encode())
    for name in ('run_extended.py','backends.py','attention_ext.py','engine.py','inversion.py','requirements-modern.txt'):
        h.update((Path(__file__).parent/name).read_bytes())
    for path in paths:h.update(Path(path).read_bytes())
    return h.hexdigest()[:20]


class ExtendedEditor:
    def __init__(self,backend,recorder,c):
        self.b,self.r,self.c=backend,recorder,c
        self.mask_sum=None;self.mask_count=0

    def capture_mask(self,grid):
        mask=self.r.effective_mask(grid,'cpu')
        if mask is not None:
            self.mask_sum=mask if self.mask_sum is None else self.mask_sum+mask
            self.mask_count+=1

    def consensus_mask(self):
        return None if self.mask_sum is None else self.mask_sum/self.mask_count

    def stream_forward(self,latent,t,embeddings,scheduler,source=False):
        profile=self.c.get('stream_profile',False) and latent.is_cuda
        if profile:torch.cuda.synchronize(latent.device)
        started=time.perf_counter()
        for attempt in range(2):
            try:
                output=(self.source_prediction(latent,t,embeddings,scheduler) if source else
                    self.b.predict(scheduler.scale_model_input(latent,t),t,embeddings))
                break
            except torch.cuda.OutOfMemoryError:
                if attempt or not self.r.stream_gpu_bytes:raise
            # Outside the exception handler: release failed-forward intermediates
            # before moving retained maps to CPU and retrying the same trajectory.
            self.r.offload_gpu_cache()
            torch.cuda.empty_cache()
            print('GPU activation space needed: replaying this step with CPU attention storage; all maps retained.',flush=True)
        if profile:torch.cuda.synchronize(latent.device)
        return output,time.perf_counter()-started

    def source_parameters(self):
        # This contract is shared by inversion, reconstruction and streaming.
        return dict(prompt=self.c['source_prompt'],negative_prompt='',guidance=1.,eta=0.)

    def target_parameters(self,prompt):
        negative=self.c['source_prompt'] if self.c.get('negative_prompt_inversion',False) else self.c.get('negative_prompt','')
        return dict(prompt=prompt,negative_prompt=negative,guidance=self.c['guidance'],eta=0.)

    def source_prediction(self,latent,t,embeddings,scheduler=None):
        scheduler=self.b.scheduler if scheduler is None else scheduler
        # No CFG batch, target embeddings, negative prompt, mask or fusion.
        return self.b.predict(scheduler.scale_model_input(latent,t),t,embeddings)

    def blend(self,edited,source,step):
        if self.c['mask_latent_blend'] and step<int(self.c['steps']*self.c.get('blend_fraction',1.)):
            weight=self.r.blend_mask(tuple(edited.shape[2:]),edited.device)
            if weight is None:raise RuntimeError('No mask available for latent preservation; select segmentation or set edit_words')
            # Masks are fp32. Keep blend arithmetic stable without promoting
            # the next diffusion input away from the model's fp16/bf16 dtype.
            return (source.float()+weight[None,None].float()*(edited.float()-source.float())).to(edited)
        return edited

    @torch.inference_mode()
    def invert(self,clean):
        latent=clean.clone()
        embeddings=self.b.embed(self.c['source_prompt'])
        iterations=1 if self.c['inversion_method']=='direct' else self.c['inversion_iterations']
        stream=self.c.get('attention_storage')=='stream'
        for i,t in enumerate(tqdm(self.b.scheduler.timesteps.flip(0),desc=f'{self.b.device} source inversion')):
            if not stream:atomic_save(latent.cpu(),self.r.root/f'latent_{i:03d}.pt',torch.save)
            previous=latent.float()
            def evaluate(guess,final):
                self.r.begin('invert' if final and not stream else 'off',i)
                output=self.source_prediction(guess.to(clean.dtype),t,embeddings).float()
                return to_epsilon(output,guess,t,self.b.scheduler)
            try:
                latent=fixed_point_solve(previous,evaluate,
                    lambda eps,anchor:inverse_ddim(eps,t,anchor,self.b.scheduler),
                    iterations,self.c['inversion_relaxation']).to(clean.dtype)
                if not torch.isfinite(latent).all():raise RuntimeError('Nonfinite inversion latent. Lower inversion_relaxation or change model precision.')
                self.r.finish()
            finally:
                self.r.mode='off';self.r.snapshot={}
        return latent

    @torch.inference_mode()
    def denoise_stream(self,noisy,prompt):
        """Advance source and edit at each timestep; retain only that step's maps.

        Both supported DDIM schedulers are stateless between step calls. The
        source path uses guidance=1, as in source reconstruction.
        """
        self.mask_sum=None;self.mask_count=0
        started=time.perf_counter()
        self.stream_performance=dict(source_forward_s=0.,target_forward_s=0.,
            cuda_synchronized=bool(self.c.get('stream_profile',False) and noisy.is_cuda),steps=0)
        source=noisy.clone();edited=noisy.clone()
        source_embeddings=self.b.embed(self.c['source_prompt'])
        conditional=self.b.embed(prompt)
        target=self.target_parameters(prompt)
        guidance=target['guidance']
        embeddings=torch.cat([self.b.embed(target['negative_prompt']),conditional]) if guidance>1 else conditional
        # Keep all scheduler state isolated even if a future implementation is stateful.
        source_scheduler=copy.deepcopy(self.b.scheduler)
        target_scheduler=copy.deepcopy(self.b.scheduler)
        self.r.align(self.b.ids(self.c['source_prompt']),self.b.ids(prompt))
        try:
            for i,t in enumerate(tqdm(self.b.scheduler.timesteps,desc=f'{self.b.device} paired source/edit')):
                try:
                    self.r.begin('source',i)
                    prediction,source_time=self.stream_forward(source,t,source_embeddings,source_scheduler,source=True)
                    self.stream_performance['source_forward_s']+=source_time
                    next_source=source_scheduler.step(prediction,t,source,eta=0.).prev_sample.to(noisy.dtype)
                    self.r.finish()  # keep current maps for the edit, without an inversion archive
                    self.r.begin('edit',i)
                    self.capture_mask(tuple(edited.shape[2:]))
                    x=torch.cat([edited,edited]) if guidance>1 else edited
                    output,target_time=self.stream_forward(x,t,embeddings,target_scheduler)
                    self.stream_performance['target_forward_s']+=target_time
                    if guidance>1:
                        unconditional,conditioned=output.chunk(2)
                        output=unconditional+guidance*(conditioned-unconditional)
                    edited=target_scheduler.step(output,t,edited,eta=0.).prev_sample.to(noisy.dtype)
                    edited=self.blend(edited,next_source,i)
                    source=next_source
                    if not torch.isfinite(source).all() or not torch.isfinite(edited).all():
                        raise RuntimeError('Nonfinite paired denoising latent')
                    self.r.finish()  # release every current-step map and spill file
                    self.stream_performance['steps']+=1
                finally:self.r.clear_stream()
        finally:
            self.r.clear_stream();self.r.mode='off'
        if self.c.get('stream_profile',False) and noisy.is_cuda:torch.cuda.synchronize(noisy.device)
        self.stream_performance['paired_total_s']=time.perf_counter()-started
        print(f'Paired timings: source forward/cache {self.stream_performance["source_forward_s"]:.2f} s; '
              f'target forward/replay {self.stream_performance["target_forward_s"]:.2f} s; '
              f'total {self.stream_performance["paired_total_s"]:.2f} s. '
              f'GPU cache peak {self.r.stats["stream_peak_gpu_bytes"]/2**30:.2f} GiB; '
              f'Attention-cache transfers D2H {self.r.stats["d2h_bytes"]/2**30:.2f} / H2D {self.r.stats["h2d_bytes"]/2**30:.2f} GiB.',flush=True)
        return edited,source

    @torch.inference_mode()
    def denoise(self,noisy,prompt,edit=False):
        self.mask_sum=None;self.mask_count=0
        z=noisy.clone()
        conditional=self.b.embed(prompt)
        guidance=self.c['guidance'] if edit else 1.
        target=self.target_parameters(prompt) if edit else self.source_parameters()
        embeddings=torch.cat([self.b.embed(target['negative_prompt']),conditional]) if guidance>1 else conditional
        if edit:self.r.align(self.b.ids(self.c['source_prompt']),self.b.ids(prompt))
        for i,t in enumerate(tqdm(self.b.scheduler.timesteps,desc=f'{self.b.device} edit' if edit else f'{self.b.device} reconstruction')):
            self.r.begin('edit' if edit else 'off',i)
            if edit:self.capture_mask(tuple(z.shape[2:]))
            x=torch.cat([z,z]) if guidance>1 else z
            output=self.b.predict(self.b.scheduler.scale_model_input(x,t),t,embeddings)
            if guidance>1:
                unconditional,conditioned=output.chunk(2)
                output=unconditional+guidance*(conditioned-unconditional)
            z=self.b.scheduler.step(output,t,z,eta=0.).prev_sample.to(noisy.dtype)
            if edit and self.c['mask_latent_blend'] and i<int(self.c['steps']*self.c.get('blend_fraction',1.)):
                anchor=torch.load(self.r.root/f'latent_{self.c["steps"]-i-1:03d}.pt',map_location=z.device,weights_only=True).to(z)
                z=self.blend(z,anchor,i)
            if not torch.isfinite(z).all():raise RuntimeError('Nonfinite denoising latent')
            self.r.finish()
        self.r.mode='off'
        return z


def source_tokens(backend,source,words):
    tokens=backend.ids(source)
    found=[]
    for word in words:
        ids=backend.tokenizer(word,add_special_tokens=False).input_ids
        for i in range(len(tokens)-len(ids)+1):
            if tokens[i:i+len(ids)]==ids:found.extend(range(i,i+len(ids)))
    return sorted(set(found))


def main(config):
    c=validate_options(normalized_config(config))
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required. CPU model inference is disabled.')
    device=torch.device('cuda:0')
    torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    torch.cuda.reset_peak_memory_stats()
    print('Backend:',c['backend'],'GPU:',torch.cuda.get_device_name(0),flush=True)
    if c.get('cache_auto_cleanup',True):cleanup_incomplete(c['cache_path'])
    out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    (out/'run_config.json').write_text(json.dumps(c,indent=2))
    paths,fps,indices=prepare_frames(c,out/'source_frames')
    c['output_fps']=fps
    images=[Image.open(p).convert('RGB') for p in paths]
    export_video(images,out/'source.mp4',fps)
    started=time.time()
    mask,mask_key=build_masks(c,indices,images,out,str(device))
    gc.collect();torch.cuda.empty_cache()
    if c.get('mask_only',False):
        print('Mask preparation complete; diffusion was not started.',flush=True)
        return
    source,revision,identity=resolve_checkpoint(c)
    key=inversion_key(c,paths,identity)
    stream=c['attention_storage']=='stream'
    with CacheManager(c['cache_path'],key,c,attention=c['cache_attention'] and not stream,latents=not stream) as cache_disk:
        root=cache_disk.path
        recorder=SourceAttention(root,c['steps'],c,mask)
        recorder.disk_manager=cache_disk
        backend=load_backend(c,recorder,source,revision)
        backend.scheduler.set_timesteps(c['steps'],device=device)
        recorder.tokens=source_tokens(backend,c['source_prompt'],c.get('edit_words',[]))
        if c['mask_backend'] in ('attention','hybrid') and c.get('edit_words') and not recorder.tokens:
            raise ValueError('edit_words were not found in source prompt tokenization')
        editor=ExtendedEditor(backend,recorder,c)
        state_path=root/'inversion.pt'
        cache_reused=False
        with torch.inference_mode():
            # Finish all text encoding before VAE/inversion/denoising. Embeddings
            # are cached on CPU; only the selected component computes on CUDA.
            prompts=[c['source_prompt'],c['target_prompt']]
            if c['guidance']>1:prompts.append(editor.target_parameters(c['target_prompt'])['negative_prompt'])
            for prompt in dict.fromkeys(prompts):backend.embed(prompt)
            state=None
            if c.get('reuse_cache',True) and cache_disk.reusable():
                try:
                    state=torch.load(state_path,map_location='cpu',weights_only=True)
                    if not torch.is_tensor(state['noisy']):raise ValueError('Invalid latent state')
                except Exception as exc:
                    print(f'Rebuilding unreadable inversion cache ({type(exc).__name__}).',flush=True)
                    cache_disk.reset();state=None
            if state is not None:
                noisy=state['noisy'].to(device=device,dtype=backend.dtype)
                cache_reused=True
                print('Reusing source inversion:',key,flush=True)
            else:
                clean=backend.encode(images)
                if not torch.isfinite(clean).all():raise RuntimeError('Video VAE produced nonfinite latents')
                if not stream and hasattr(backend,'prepare_attention_cache'):backend.prepare_attention_cache(clean)
                noisy=editor.invert(clean)
                atomic_save({'noisy':noisy.cpu()},state_path,torch.save)
                cache_disk.mark_complete()
                del clean
            reconstruction=None
            if stream:
                # Preparation is needed on reused noisy latents too: only their
                # shape matters for the current denoising cache plan.
                if hasattr(backend,'prepare_attention_cache'):backend.prepare_attention_cache(noisy)
                edited_latents,source_latents=editor.denoise_stream(noisy,c['target_prompt'])
                if c.get('reconstruct',False):
                    reconstruction=backend.decode(source_latents)[:len(images)]
                    export_video(reconstruction,out/'reconstruction.mp4',fps)
                result=backend.decode(edited_latents)[:len(images)]
            elif c.get('reconstruct',False):
                reconstruction=backend.decode(editor.denoise(noisy,c['source_prompt']))[:len(images)]
                export_video(reconstruction,out/'reconstruction.mp4',fps)
            if not stream:result=backend.decode(editor.denoise(noisy,c['target_prompt'],edit=True))[:len(images)]
        if len(result)!=len(images):raise RuntimeError('Native decoder frame count does not match source')
        if c['mask_composite']:
            # Explicit optional pixel compositing, independently selectable from diffusion masking.
            raw=out/'edited_uncomposited_frames';raw.mkdir(exist_ok=True)
            for i,image in enumerate(result):image.save(raw/f'{i:05d}.png')
            result=[Image.composite(edit,src,Image.fromarray((m[0].numpy()*255).round().astype('uint8')))
                    for edit,src,m in zip(result,images,mask)]
        # Export the mask actually selected for editing, including attention masks.
        saved_mask=mask if c['mask_backend'] not in ('attention','hybrid') else editor.consensus_mask()
        masks_dir=out/'masks';masks_dir.mkdir(exist_ok=True)
        for old in masks_dir.glob('[0-9][0-9][0-9][0-9][0-9].png'):old.unlink()
        if saved_mask is not None:
            if saved_mask.ndim==3:saved_mask=saved_mask[:,None]
            full=mask_grid(saved_mask,(len(images),c['size'],c['size'])).cpu()
            for i,value in enumerate(full):
                Image.fromarray((value.numpy().clip(0,1)*255).round().astype('uint8')).save(masks_dir/f'{i:05d}.png')
        frames=out/'edited_frames';frames.mkdir(exist_ok=True)
        for old in frames.glob('[0-9][0-9][0-9][0-9][0-9].png'):old.unlink()
        for i,image in enumerate(result):image.save(frames/f'{i:05d}.png')
        export_video(result,out/'edited.mp4',fps)
        comparisons=[]
        for i,src in enumerate(images):
            panels=[src]+([reconstruction[i]] if reconstruction else [])+[result[i]]
            canvas=Image.new('RGB',(c['size']*len(panels),c['size']))
            for j,panel in enumerate(panels):canvas.paste(panel,(j*c['size'],0))
            comparisons.append(canvas)
        export_video(comparisons,out/'comparison.mp4',fps)
        comparisons[0].save(out/'comparison_first_frame.png')
        change=np.stack([np.abs(np.asarray(a).astype(float)-np.asarray(b).astype(float)) for a,b in zip(images,result)])
        report=dict(config=c,implementation='optional-backends-v4-stream',resolved_model_revision=identity,
            prediction_type=backend.scheduler.config.prediction_type,source_frame_indices=indices,fps=fps,
            runtime_seconds=time.time()-started,peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,
            cache_key=key,cache_reused=cache_reused,mask_cache_key=mask_key,
            attention_reference='source_denoising' if stream else 'source_inversion',
            source_conditioning=editor.source_parameters(),target_conditioning=editor.target_parameters(c['target_prompt']),
            image_adaptation_applied=c['backend']!='cogvideox_2b',
            effective_inversion_iterations=1 if c['inversion_method']=='direct' else c['inversion_iterations'],
            unet_calls=backend.calls,unet_device=str(device),attention_stats=recorder.stats,
            edit_mean_absolute_pixel_change=float(change.mean()),
            comparison_order=['source','reconstruction','edit'] if reconstruction else ['source','edit'])
        if stream:report['stream_performance']=editor.stream_performance
        if mask is not None:
            outside=1-mask[:,0].numpy()
            report['outside_mask_mean_absolute_pixel_change']=float((change*outside[...,None]).sum()/max(3*outside.sum(),1))
        if reconstruction:
            mse=np.mean([(np.asarray(a).astype(float)-np.asarray(b).astype(float))**2 for a,b in zip(images,reconstruction)])
            report['reconstruction_psnr_db']=10*math.log10(255**2/max(mse,1e-10))
        report['cache_evicted_runs']=cache_disk.removed
        (out/'run_report.json').write_text(json.dumps(report,indent=2))
        if not c.get('keep_cache',False):cache_disk.release()
        print('Saved outputs:',out,flush=True)
