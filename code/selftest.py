"""No-download checks. The default tiny model test REQUIRES a CUDA GPU."""
import argparse
import tempfile
from pathlib import Path
import torch
from unittest.mock import patch
from test_inversion import FixedPointTests
from engine import VideoUNet, AttentionCache, Editor, align_tokens, inverse_step


def checks(device):
    from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTextConfig
    torch.manual_seed(0)
    torch.set_num_threads(2)
    mapper, matches = align_tokens([0, 2, 3, 9], [0, 4, 3, 9])
    assert matches.tolist() == [True, False, True, True]
    assert mapper.tolist() == [0, 1, 2, 3]
    with tempfile.TemporaryDirectory() as directory:
        cache = AttentionCache(directory, 2, 1., 1.)
        cache.mapper, cache.matched = mapper, matches
        source = torch.rand(2, 2, 4, 4, device=device).softmax(-1)
        cache.begin('invert', 0)
        cache.apply('block.attn2', source, True, 2)
        cache.apply('block.attn1', source, False, 2)
        cache.finish()
        cache.begin('invert', 1)
        cache.apply('block.attn2', source, True, 2)
        cache.apply('block.attn1', source, False, 2)
        cache.finish()
        cache.begin('edit', 0)
        target = torch.rand(4, 2, 4, 4, device=device)
        before = target.clone()
        after = cache.apply('block.attn2', target, True, 2)
        torch.testing.assert_close(after[:2], before[:2])
        torch.testing.assert_close(after[-2:, ..., 1], before[-2:, ..., 1])
        torch.testing.assert_close(after[-2:, ..., 2], source[..., 2])
        after = cache.apply('block.attn1', before.clone(), False, 2)
        torch.testing.assert_close(after[-2:], source)
        print('PASS reverse disk cache, CFG isolation, target colour-token attention, source self fusion')

        backbone = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
            down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
            up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'), block_out_channels=(32, 64),
            layers_per_block=1, norm_num_groups=8, cross_attention_dim=32, attention_head_dim=4)
        unet = VideoUNet(backbone, rank=4, least_channel=0).to(device).eval().requires_grad_(False)
        vae = AutoencoderKL(in_channels=3, out_channels=3,
            down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D'),
            up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D'), block_out_channels=(32,64),
            layers_per_block=1, latent_channels=4, norm_num_groups=8, sample_size=16).to(device).eval()
        text = CLIPTextModel(CLIPTextConfig(vocab_size=12, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=77)).to(device).eval()
        class Tokens:
            def __init__(self, ids): self.input_ids = ids
        class Tokenizer:
            def __call__(self, prompt, return_tensors=None, **kwargs):
                ids = [0, 2 if 'pink' in prompt else 3, 4] + [1]*74
                return Tokens(torch.tensor([ids]) if return_tensors else ids)
        scheduler = DDIMScheduler(num_train_timesteps=100, clip_sample=False, set_alpha_to_one=False, steps_offset=1)
        scheduler.set_timesteps(2, device=device)
        editor = Editor(unet, vae, text, Tokenizer(), scheduler, cache)
        with torch.inference_mode():
            clean = torch.randn(1,4,2,8,8, device=device)
            model_modes = []
            handle = unet.register_forward_pre_hook(lambda module, args: model_modes.append(cache.mode))
            with patch('engine.torch.save', wraps=torch.save) as save:
                noisy = editor.invert(clean, 'pink flowers', iterations=3)
            handle.remove()
            assert model_modes == ['off', 'off', 'invert'] * 2
            assert save.call_count == 2  # one snapshot per timestep, never per iteration
            assert cache.mode == 'off' and cache.snapshot == {}
            print('PASS fixed-point UNet iteration count and final-iteration-only attention snapshots')
            reconstructed = editor.denoise(noisy, 'pink flowers', guidance=1.)
            edited = editor.denoise(noisy, 'blue flowers', 'pink flowers')
            assert edited.device == clean.device and torch.isfinite(edited).all()
            assert not torch.equal(edited, reconstructed)
            frames = editor.decode(edited)
            assert len(frames) == 2 and frames[0].size == (16,16)
            assert unet.calls == 10
            cache.mask_tokens = [2]
            cache.begin('edit', 0)
            assert cache.edit_mask.shape == (2,1,32,32)
        print(f'PASS tiny custom video UNet, inversion, reconstruction, edit, mask and VAE on {device}')
    print('Random-weight checks do not establish pretrained visual quality or full-model VRAM use.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--development-cpu', action='store_true', help='Explicit local developer wiring check only')
    args = parser.parse_args()
    if not args.development_cpu and not torch.cuda.is_available():
        raise RuntimeError('GPU required: enable a Kaggle GPU accelerator. No CPU fallback.')
    device = torch.device('cpu' if args.development_cpu else 'cuda:0')
    if device.type == 'cuda': print('GPU:', torch.cuda.get_device_name(0))
    import unittest
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(FixedPointTests))
    if not result.wasSuccessful():
        raise RuntimeError('Fixed-point numerical checks failed')
    checks(device)
