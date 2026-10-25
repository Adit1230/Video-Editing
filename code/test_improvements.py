"""Regression checks for conditioning isolation, blending and step-cache lifetime."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from diffusers import DDIMScheduler,CogVideoXDDIMScheduler
from options import normalized_config,validate_options,EDIT_PROFILES
from attention_ext import SourceAttention
from run_extended import ExtendedEditor


class Backend:
    def __init__(self,recorder,scheduler):
        self.recorder,self.scheduler=recorder,scheduler
        self.device=torch.device('cpu');self.calls=[]
    def ids(self,prompt):return [1,2]
    def embed(self,prompt):return torch.tensor([{'source':.1,'target':.3,'other':-.2,'bad':-.4,'':0.}[prompt]])
    def predict(self,z,t,embeddings):
        self.calls.append((self.recorder.mode,z.shape[0],embeddings.clone()))
        if self.recorder.mode=='invert':self.recorder.put('step',torch.tensor([self.recorder.step]))
        if self.recorder.mode=='edit':
            assert self.recorder.value('step').item()==self.recorder.step
            assert len(self.recorder.snapshot)==1
        return .01*z+embeddings.reshape(-1,1,1,1,1).to(z)


class Improvements(unittest.TestCase):
    def config(self,**overrides):
        c=dict(backend='sd21',source_prompt='source',target_prompt='target',steps=4,size=128,frames=2,
            guidance=4.,negative_prompt='bad',attention_storage='stream',mask_backend='file',mask_dir='/tmp/masks',
            stream_cache_ram_gib=1e-12,edit_words=[])
        c.update(overrides)
        return normalized_config(c)

    def test_source_contract_and_spills(self):
        for scheduler_type in (DDIMScheduler,CogVideoXDDIMScheduler):
            for prediction_type in ('epsilon','v_prediction'):
                for negative_inversion in (False,True):
                    with self.subTest(scheduler=scheduler_type.__name__,prediction=prediction_type,negative=negative_inversion),tempfile.TemporaryDirectory() as tmp:
                        c=self.config(negative_prompt_inversion=negative_inversion,mask_latent_blend=True)
                        r=SourceAttention(Path(tmp),4,c,torch.zeros(2,1,8,8))
                        scheduler=scheduler_type(num_train_timesteps=100,clip_sample=False,prediction_type=prediction_type)
                        scheduler.set_timesteps(4)
                        b=Backend(r,scheduler);editor=ExtendedEditor(b,r,c)
                        clean=torch.ones(1,2,2,8,8)
                        noisy=editor.invert(clean)
                        assert all(batch==1 and torch.equal(emb,b.embed('source')) for _,batch,emb in b.calls)
                        assert not list(Path(tmp).glob('*.pt'))
                        expected_source=editor.denoise(noisy,'source')
                        b.calls=[]
                        result,source=editor.denoise_stream(noisy,'target')
                        torch.testing.assert_close(source,expected_source)
                        torch.testing.assert_close(result,source)  # all-background mask
                        source_calls=b.calls[::2];target_calls=b.calls[1::2]
                        assert all(batch==1 and torch.equal(emb,b.embed('source')) for _,batch,emb in source_calls)
                        negative='source' if negative_inversion else 'bad'
                        assert all(batch==2 and torch.equal(emb,torch.cat([b.embed(negative),b.embed('target')])) for _,batch,emb in target_calls)
                        assert editor.source_parameters()==dict(prompt='source',negative_prompt='',guidance=1.,eta=0.)
                        c.update(guidance=9.,negative_prompt='other',negative_prompt_inversion=not negative_inversion,mask_latent_blend=False)
                        _,again=editor.denoise_stream(noisy,'other')
                        torch.testing.assert_close(source,again)
                        assert not r.snapshot and not r.stream_active and not list(Path(tmp).glob('*.pt'))
                        assert r.stats['stream_spilled_bytes']>0

    def test_cache_oom_retry_does_not_advance_trajectories(self):
        # Inject a CUDA-OOM exception on CPU to test retry/scheduler control flow;
        # actual CUDA map offload is covered by the GPU storage smoke test.
        for phase in ('invert','edit'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as tmp:
                c=self.config(mask_latent_blend=True)
                mask=torch.zeros(2,1,8,8)
                scheduler=DDIMScheduler(num_train_timesteps=100,clip_sample=False)
                scheduler.set_timesteps(4)
                r=SourceAttention(Path(tmp),4,c,mask)
                b=Backend(r,scheduler);editor=ExtendedEditor(b,r,c)
                noisy=torch.ones(1,2,2,8,8)
                expected,expected_source=editor.denoise_stream(noisy,'target')
                original=b.predict;failed=False
                def fail_once(*args):
                    nonlocal failed
                    output=original(*args)
                    if not failed and r.mode==phase and r.step==1:
                        failed=True;r.stream_gpu_bytes=1
                        raise torch.cuda.OutOfMemoryError('simulated cached-map activation pressure')
                    return output
                original_offload=r.offload_gpu_cache
                def offload():
                    original_offload();r.stream_gpu_bytes=0
                with patch.object(b,'predict',side_effect=fail_once),patch.object(r,'offload_gpu_cache',side_effect=offload),patch('torch.cuda.empty_cache'):
                    actual,source=editor.denoise_stream(noisy,'target')
                torch.testing.assert_close(actual,expected)
                torch.testing.assert_close(source,expected_source)
                assert failed and r.stats['gpu_cache_oom_fallbacks']==1
                assert r.gpu_cache_limit==0 and not r.snapshot and not list(Path(tmp).glob('*.pt'))

    def test_backend_dtype_boundaries(self):
        from backends import ImageBackend,CogVideoBackend
        class Denoiser(torch.nn.Module):
            def __init__(self,native):
                super().__init__();self.native=native
                self.projection=torch.nn.Linear(3,3).half()
                self.context=SimpleNamespace(frames=1)
                self.config=SimpleNamespace(patch_size=1,use_rotary_positional_embeddings=False)
            def forward(self,x=None,t=None,encoder_hidden_states=None,hidden_states=None,**kwargs):
                z=hidden_states if self.native else x
                assert z.dtype==encoder_hidden_states.dtype==torch.float16
                # A real half-weight Linear reproduces the failing dtype constraint.
                self.projection(encoder_hidden_states)
                return (z,) if self.native else SimpleNamespace(sample=z)
        for native in (False,True):
            b=(CogVideoBackend if native else ImageBackend).__new__(CogVideoBackend if native else ImageBackend)
            b.device=torch.device('cpu');b.dtype=torch.float16;b.calls=0;b.rope=None
            b.stages=SimpleNamespace(activate=lambda name:None)
            b.recorder=SimpleNamespace(grid=None)
            model=Denoiser(native)
            b.pipe=SimpleNamespace(**({'transformer':model} if native else {'unet':model}))
            result=b.predict(torch.ones(1,4,2,8,8),torch.tensor(1),torch.ones(1,2,3))
            assert result.dtype==torch.float16 and result.shape==(1,4,2,8,8)

    def test_half_precision_blend_and_next_step(self):
        class HalfBackend(Backend):
            def embed(self,prompt):return super().embed(prompt).half()
            def predict(self,z,t,embeddings):
                # Reproduce the strict half-weight Linear input requirement.
                assert z.dtype==embeddings.dtype==torch.float16, (z.dtype,embeddings.dtype)
                return super().predict(z,t,embeddings)
        for mask_mode in ('attention','file','hybrid'):
            for dtype in (torch.float16,torch.bfloat16):
                with self.subTest(mask=mask_mode,dtype=dtype),tempfile.TemporaryDirectory() as tmp:
                    c=self.config(mask_backend=mask_mode,mask_latent_blend=True,blend_dilate=1,blend_feather=1)
                    mask=torch.zeros(2,1,8,8);mask[:,:,:,4:]=1
                    r=SourceAttention(Path(tmp),4,c,mask)
                    r.heat_mask=mask.clone()
                    editor=ExtendedEditor(None,r,c)
                    z=torch.ones(1,2,2,8,8,dtype=dtype);source=torch.zeros_like(z)
                    result=editor.blend(z,source,0)
                    assert result.dtype==dtype and result.device==z.device
                    assert ((result>0)&(result<1)).any()
                    torch.testing.assert_close(source,torch.zeros_like(source))
                    if mask_mode!='file' or dtype!=torch.float16:continue
                    scheduler=DDIMScheduler(num_train_timesteps=100,clip_sample=False)
                    scheduler.set_timesteps(4)
                    backend=HalfBackend(r,scheduler);editor=ExtendedEditor(backend,r,c)
                    # Four full paired steps expose the previous iteration-2 crash.
                    result,reference=editor.denoise_stream(z,'target')
                    assert result.dtype==reference.dtype==torch.float16
                    assert len(backend.calls)==8 and not r.snapshot
                    # Disk replay uses the same blend operation and must also stay half.
                    for i in range(4):
                        torch.save({'step':torch.tensor([i])},Path(tmp)/f'{3-i:03d}.pt')
                        torch.save(source,Path(tmp)/f'latent_{3-i:03d}.pt')
                    result=editor.denoise(z,'target',edit=True)
                    assert result.dtype==torch.float16

    def test_blend_schedule_and_segmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=self.config(mask_latent_blend=True,blend_fraction=.5,blend_dilate=1,blend_feather=1)
            mask=torch.zeros(9,1,8,8);mask[4:,:,3:5,3:5]=1
            r=SourceAttention(Path(tmp),4,c,mask)
            editor=ExtendedEditor(None,r,c)
            source=torch.zeros(1,2,3,8,8);edited=torch.ones_like(source)
            weight=r.blend_mask((3,8,8),'cpu')
            assert weight.shape==(3,8,8) and ((weight>0)&(weight<1)).any()
            torch.testing.assert_close(editor.blend(edited,source,0),weight[None,None].expand_as(edited))
            torch.testing.assert_close(editor.blend(edited,source,2),edited)

    def test_profiles_aliases_validation(self):
        c=self.config(edit_profile='improved')
        assert c['negative_prompt_inversion'] and c['zero_temporal_bias'] and c['sparse_least_channel']==320
        validate_options(c)
        assert normalized_config(c)==c
        assert self.config(latent_blend=True)['mask_latent_blend']
        for override in (dict(blend_fraction=1.5),dict(blend_dilate=-1),dict(zero_temporal_bias=1),dict(sparse_least_channel=123)):
            with self.assertRaises(ValueError):validate_options(self.config(**override))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
