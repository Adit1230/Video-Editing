"""Streaming performance contracts. CUDA by default; explicit developer CPU mode."""
import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from diffusers.models.attention_processor import Attention
from attention_ext import SourceAttention,JointVideoProcessor,bounded_sdpa,mask_grid,stream_gpu_capacity
from backends import ImageAttentionProcessor
from options import normalized_config


def config(**kw):
    c=dict(backend='sd21',attention_storage='stream',self_replace=.5,cross_replace=.25,
        mask_backend='attention',attention_heads=0,attention_max_queries=128,attention_query_chunk=8,
        stream_cache_device='auto',stream_gpu_reserve_gib=0.,stream_cache_gpu_gib=0.,
        source_prompt='a cat',target_prompt='a dog')
    c.update(kw);return normalized_config(c)


def storage_check(root,device):
    value=torch.arange(64,device=device,dtype=torch.float32).reshape(8,8)
    size=value.numel()*value.element_size()
    c=config(stream_cache_gpu_gib=1.5*size/2**30)
    r=SourceAttention(root/'storage',4,c)
    r.begin('source',0);r.put('first',value);r.put('second',value+1)
    expected=value.clone();value.fill_(0)
    torch.testing.assert_close(r.value('first'),expected)
    if device.type=='cuda':
        assert r.value('first').is_cuda and not r.value('second').is_cuda
        assert r.stream_gpu_bytes==size and r.stats['d2h_bytes']==size
        torch.testing.assert_close(r.source_value('first',expected),expected)
        assert r.stats['h2d_bytes']==0
        torch.testing.assert_close(r.source_value('second',expected),expected+1)
        assert r.stats['h2d_bytes']==size
        r.offload_gpu_cache()
        assert not r.stream_gpu_bytes and r.stats['gpu_cache_oom_fallbacks']==1
        torch.testing.assert_close(r.value('first').to(device),expected)
    else:assert not r.stream_gpu_bytes and r.stream_ram_bytes==2*size
    r.clear_stream();assert not r.snapshot and not list(r.root.glob('*.pt'))
    c=config(stream_cache_device='cpu',stream_cache_ram_gib=1e-12)
    r=SourceAttention(root/'spill',4,c);r.begin('source',0);r.put('map',expected)
    assert isinstance(r.snapshot['map'],Path) and r.stats['stream_spilled_bytes']==size
    torch.testing.assert_close(r.value('map').to(device),expected)
    # Replacing a recorded entry on retry must not accumulate phantom cache size.
    r.put('map',expected+1);assert r.stream_bytes==size
    torch.testing.assert_close(r.value('map').to(device),expected+1)
    r.clear_stream();assert not list(r.root.glob('*.pt'))
    print('PASS GPU/RAM/disk storage selection, immutable copies, transfers, offload and retry accounting')


def schedule_check(root,device):
    c=config();r=SourceAttention(root/'image',4,c);r.tokens=[1]
    owner=SimpleNamespace(context=SimpleNamespace(frames=2),least_channel=0,recorder=r)
    attn=Attention(query_dim=32,heads=4,dim_head=8).to(device).eval()
    x=torch.randn(2,16,32,device=device);text=torch.randn(2,7,32,device=device)
    own=ImageAttentionProcessor(owner,'self');cross=ImageAttentionProcessor(owner,'cross')
    for step in range(4):
        r.begin('source',step)
        own(attn,x);cross(attn,x,text)
        keys=list(r.snapshot)
        assert any(name.startswith('self/q') for name in keys)==(step<2)
        assert any(name.startswith('cross/q') for name in keys)==(step<1)
        assert 'cross/heat' in keys
        r.finish();r.begin('edit',step)
        r.align(list(range(7)),list(range(7)))
        with patch('backends.F.interpolate',wraps=torch.nn.functional.interpolate) as resize:
            a=r.effective_mask((2,4,4),device);b=r.effective_mask((2,4,4),device)
            assert a is b and resize.call_count==1
        if step>=2:
            with patch.object(r,'image',wraps=r.image) as replay:
                own(attn,x);cross(attn,x,text)
                assert replay.call_count==0
        r.finish();assert not r.snapshot
    # Segmentation runs do not need cross-attention heat maps.
    r.c['mask_backend']='file';r.begin('source',3)
    own(attn,x);cross(attn,x,text);assert not r.snapshot
    r.clear_stream()
    # Native joint rows have the same fusion lifecycle; mask heat still survives.
    c=config(backend='cogvideox_2b',attention_layers=[0])
    r=SourceAttention(root/'native',4,c);r.grid=(2,2,2);r.tokens=[1]
    processor=JointVideoProcessor(r,0)
    x=torch.randn(1,8,32,device=device);text=torch.randn(1,7,32,device=device)
    for step in (0,2,3):
        r.begin('source',step);processor(attn,x,text)
        assert any('/q' in name for name in r.snapshot)==(step==0)
        assert 'joint/0/heat' in r.snapshot
        r.clear_stream()
    print('PASS active-window-only maps and replay, retained heat masks, mask reuse, image/native attention')


def capacity_and_sdpa_check(device):
    gib=2**30
    with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.current_device',return_value=0), \
         patch('torch.cuda.mem_get_info',return_value=(8*gib,16*gib)), \
         patch('torch.cuda.memory_reserved',return_value=3*gib),patch('torch.cuda.memory_allocated',return_value=2*gib):
        assert stream_gpu_capacity(config(stream_gpu_reserve_gib=4.))==5*gib
        assert stream_gpu_capacity(config(stream_gpu_reserve_gib=4.,stream_cache_gpu_gib=2.))==2*gib
        assert stream_gpu_capacity(config(stream_cache_device='cpu'))==0
        with tempfile.TemporaryDirectory() as tmp:
            r=SourceAttention(tmp,4,config(stream_gpu_reserve_gib=4.))
            r.begin('source',0);r.stream_gpu_bytes=gib//2;r.gpu_cache_limit=gib
            assert r.gpu_room(SimpleNamespace(is_cuda=True,device=0))==gib//2
            r.clear_stream()
    q=torch.randn(2,4,33,8,device=device);k=torch.randn(2,4,17,8,device=device);v=torch.randn_like(k)
    expected=torch.nn.functional.scaled_dot_product_attention(q,k,v)
    torch.testing.assert_close(bounded_sdpa(q,k,v,8),expected,atol=1e-6,rtol=1e-5)
    print('PASS VRAM headroom/cap policy and fused attention equivalence')


def main(device):
    torch.set_num_threads(2);torch.manual_seed(5)
    with torch.inference_mode(),tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);storage_check(root,device);schedule_check(root,device);capacity_and_sdpa_check(device)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--development-cpu',action='store_true');args=parser.parse_args()
    if not args.development_cpu and not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU checks need --development-cpu')
    main(torch.device('cpu' if args.development_cpu else 'cuda:0'))
