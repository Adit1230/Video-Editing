"""Independent Video Editing-method implementation built on Diffusers SD1.x layers.

No Video Editing source imports. Video tensors are flattened only at the boundary;
video-aware convolution, residual normalization and attention preserve frame axes.
"""
from pathlib import Path
from types import MethodType
import math
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from inversion import fixed_point_solve, validate_fixed_point
from cache_manager import atomic_save


class VideoContext:
    frames = 1


class TemporalConv(nn.Module):
    def __init__(self, spatial, context, rank=160):
        super().__init__()
        self.spatial, self.context = spatial, context
        channels = spatial.out_channels
        rank = rank if rank <= channels else channels // 2
        self.down = nn.Conv1d(channels, rank, 3, padding=1, bias=False)
        self.up = nn.Conv1d(rank, channels, 3, padding=1, bias=False)
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        y = self.spatial(x)
        bf, c, h, w = y.shape
        f = self.context.frames
        temporal = y.reshape(bf // f, f, c, h, w).permute(0, 3, 4, 2, 1).reshape(-1, c, f)
        temporal = temporal + self.up(self.down(temporal))
        return temporal.reshape(bf // f, h, w, c, f).permute(0, 4, 3, 1, 2).reshape_as(y)


class VideoGroupNorm(nn.Module):
    def __init__(self, norm, context):
        super().__init__()
        self.norm, self.context = norm, context

    def forward(self, x):
        bf, c, h, w = x.shape
        f = self.context.frames
        x = x.reshape(bf // f, f, c, h, w).transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2).reshape(bf, c, h, w)


def align_tokens(source, target):
    """Global sequence alignment, mismatch cost 2 and gap cost 1."""
    n, m = len(source), len(target)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): cost[i][0] = i
    for j in range(m + 1): cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i-1][j] + 1, cost[i][j-1] + 1,
                             cost[i-1][j-1] + (0 if source[i-1] == target[j-1] else 2))
    mapper, matched = list(range(m)), [False] * m
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i-1][j-1] + (0 if source[i-1] == target[j-1] else 2):
            mapper[j-1], matched[j-1] = i-1, source[i-1] == target[j-1]
            i, j = i-1, j-1
        elif j and cost[i][j] == cost[i][j-1] + 1:
            mapper[j-1] = max(0, i-1)
            j -= 1
        else: i -= 1
    return torch.tensor(mapper), torch.tensor(matched)


class AttentionCache:
    def __init__(self, root, steps, self_fraction=.5, cross_fraction=.3):
        self.root, self.steps = Path(root), steps
        self.root.mkdir(parents=True, exist_ok=True)
        self.self_fraction, self.cross_fraction = self_fraction, cross_fraction
        self.mode, self.step, self.snapshot = 'off', 0, {}
        self.disk_manager = None
        self.snapshot_bytes = 0
        self.mapper = self.matched = None
        self.edit_mask = None
        self.mask_tokens = []
        self.mask_threshold = .3
        self.stats = {'stored_layers': 0, 'fused_self_layers': 0, 'fused_cross_layers': 0}

    def begin(self, mode, step):
        self.mode, self.step = mode, step
        self.snapshot = torch.load(self.root / f'{self.steps-step-1:03d}.pt', map_location='cpu') if mode == 'edit' else {}
        self.edit_mask = None
        if mode == 'edit' and self.mask_tokens:
            maps = []
            for name, value in self.snapshot.items():
                if name.endswith('attn2'):
                    size = math.isqrt(value.shape[-2])
                    if size * size != value.shape[-2]: continue
                    heat = value[..., self.mask_tokens].float().sum(-1).mean(1)
                    heat = heat.reshape(-1, 1, size, size)
                    maps.append(F.interpolate(heat, size=(32, 32), mode='bilinear', align_corners=False))
            if maps:
                heat = torch.stack(maps).mean(0)
                heat = F.max_pool2d(heat, 3, stride=1, padding=1)
                self.edit_mask = heat / heat.amax((-1, -2), keepdim=True).clamp_min(1e-8) > self.mask_threshold

    def finish(self):
        if self.mode == 'invert':
            path = self.root / f'{self.step:03d}.pt'
            if self.disk_manager is not None:
                self.disk_manager.ensure_space(self.snapshot_bytes)
            atomic_save(self.snapshot, path, torch.save)
            self.snapshot_bytes = path.stat().st_size
            if self.step == 0:
                import shutil
                estimate = path.stat().st_size * self.steps
                print(f'Attention cache estimate: {estimate / 2**30:.2f} GiB', flush=True)
                remaining = self.snapshot_bytes * (self.steps - 1)
                if self.disk_manager is not None:
                    self.disk_manager.ensure_space(remaining)
                elif remaining + 2 * 2**30 > shutil.disk_usage(self.root).free:
                    raise RuntimeError('Not enough disk for the remaining attention cache and 2 GiB reserve.')
        self.snapshot = {}

    def apply(self, name, probs, cross, frames):
        # probs: CFG batch*frames, heads, spatial queries, keys.
        if self.mode == 'invert':
            self.snapshot[name] = probs.detach().to('cpu', copy=True)
            self.stats['stored_layers'] += 1
        elif self.mode == 'edit':
            source = self.snapshot[name].to(probs)
            conditional = probs[-frames:]
            if cross and self.step < int(self.steps * self.cross_fraction):
                mapper = self.mapper.to(probs.device)
                mask = self.matched.to(probs.device).view(1, 1, 1, -1)
                probs[-frames:] = torch.where(mask, source[..., mapper], conditional)
                self.stats['fused_cross_layers'] += 1
            elif not cross and self.step < int(self.steps * self.self_fraction):
                if self.edit_mask is None:
                    probs[-frames:] = source
                else:
                    size = math.isqrt(probs.shape[-2])
                    mask = F.interpolate(self.edit_mask.float(), size=(size, size), mode='nearest')
                    mask = mask.to(probs.device).reshape(frames, 1, -1, 1)
                    probs[-frames:] = mask * conditional + (1-mask) * source
                self.stats['fused_self_layers'] += 1
        return probs


class VideoUNet(nn.Module):
    """SD1.x spatial backbone plus temporal residual convs and attention.

    Rank=160, middle-frame K/V at channels>=640, temporal attention after FF.
    Diffusers supplies standard pretrained SD layers, not Video Editing code.
    """
    def __init__(self, backbone, rank=160, least_channel=640):
        super().__init__()
        from diffusers.models.attention import CrossAttention
        self.backbone = backbone
        self.context = VideoContext()
        self.cache = None
        self.least_channel = least_channel
        self.calls = 0
        self.last_device = None
        # Materialize lists before replacement to avoid visiting newly added layers.
        for name, module in list(backbone.named_modules()):
            if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
                parent_name, _, child = name.rpartition('.')
                parent = backbone.get_submodule(parent_name) if parent_name else backbone
                setattr(parent, child, TemporalConv(module, self.context, rank))
            elif isinstance(module, nn.GroupNorm) and ('.resnets.' in name or name == 'conv_norm_out'):
                parent_name, _, child = name.rpartition('.')
                parent = backbone.get_submodule(parent_name) if parent_name else backbone
                setattr(parent, child, VideoGroupNorm(module, self.context))
        for name, module in list(backbone.named_modules()):
            if module.__class__.__name__ == 'BasicTransformerBlock':
                dim = module.norm1.normalized_shape[0]
                module.norm_temporal = nn.LayerNorm(dim)
                module.attn_temporal = CrossAttention(query_dim=dim, heads=module.attn1.heads,
                    dim_head=dim // module.attn1.heads, bias=False)
                nn.init.zeros_(module.attn_temporal.to_out[0].weight)
                owner = self
                original = module.forward
                def temporal_forward(block, *args, _forward=original, **kwargs):
                    y = _forward(*args, **kwargs)
                    bf, spatial, channels = y.shape
                    f = owner.context.frames
                    z = y.reshape(bf // f, f, spatial, channels).transpose(1, 2).reshape(-1, f, channels)
                    z = z + block.attn_temporal(block.norm_temporal(z))
                    return z.reshape(bf // f, spatial, f, channels).transpose(1, 2).reshape_as(y)
                module.forward = MethodType(temporal_forward, module)
        for name, module in list(backbone.named_modules()):
            if isinstance(module, CrossAttention):
                owner = self
                def attention(attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                              _name=name, **kwargs):
                    if attention_mask is not None: raise ValueError('Masked text inputs are unsupported')
                    cross = encoder_hidden_states is not None
                    context = hidden_states if not cross else encoder_hidden_states
                    q, k, v = attn.to_q(hidden_states), attn.to_k(context), attn.to_v(context)
                    batch, queries, dim = q.shape
                    heads, f = attn.heads, owner.context.frames
                    temporal = _name.endswith('attn_temporal')
                    if not cross and not temporal and dim >= owner.least_channel:
                        def middle(x):
                            x = x.reshape(batch // f, f, x.shape[1], dim)
                            return x[:, (f-1)//2:(f-1)//2+1].expand(-1, f, -1, -1).reshape(batch, -1, dim)
                        k, v = middle(k), middle(v)
                    def split(x): return x.reshape(batch, x.shape[1], heads, dim // heads).transpose(1, 2)
                    q, k, v = split(q), split(k), split(v)
                    if not temporal and queries <= 1024 and owner.cache and owner.cache.mode != 'off':
                        probs = (q @ k.transpose(-1, -2) * attn.scale).softmax(-1)
                        probs = owner.cache.apply(_name, probs, cross, f)
                        y = probs @ v
                    else:
                        # SDPA chooses a fused CUDA backend when supported; chunked
                        # fallback bounds memory on P100 without empty_cache per step.
                        y = torch.empty_like(q)
                        for b in range(0, batch, 4):
                            for start in range(0, queries, 512):
                                y[b:b+4, :, start:start+512] = F.scaled_dot_product_attention(
                                    q[b:b+4, :, start:start+512], k[b:b+4], v[b:b+4], dropout_p=0.)
                    y = y.transpose(1, 2).reshape(batch, queries, dim)
                    return attn.to_out[1](attn.to_out[0](y))
                module.forward = MethodType(attention, module)

    def forward(self, latents, timestep, embeddings):
        b, c, f, h, w = latents.shape
        self.context.frames = f
        self.calls += 1
        self.last_device = str(latents.device)
        x = latents.permute(0, 2, 1, 3, 4).reshape(b*f, c, h, w)
        y = self.backbone(x, timestep, encoder_hidden_states=embeddings.repeat_interleave(f, 0)).sample
        return y.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4)


def inverse_step(noise, timestep, latent, scheduler):
    t = int(timestep)
    previous = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    alpha = scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.final_alpha_cumprod
    next_alpha = scheduler.alphas_cumprod[t]
    alpha, next_alpha = alpha.to(latent), next_alpha.to(latent)
    clean = (latent - (1-alpha).sqrt() * noise) / alpha.sqrt()
    return next_alpha.sqrt() * clean + (1-next_alpha).sqrt() * noise


class Editor:
    def __init__(self, unet, vae, text_encoder, tokenizer, scheduler, cache):
        self.unet, self.vae, self.text_encoder = unet, vae, text_encoder
        self.tokenizer, self.scheduler, self.cache = tokenizer, scheduler, cache
        unet.cache = cache
        self.device = next(unet.parameters()).device

    def embed(self, prompt):
        tokens = self.tokenizer(prompt, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
        return self.text_encoder(tokens.input_ids.to(self.device))[0]

    @torch.inference_mode()
    def invert(self, clean, prompt, iterations=3, relaxation=1.):
        validate_fixed_point(iterations, relaxation)
        latent = clean.clone()
        embeddings = self.embed(prompt)
        print(f'Fixed-point inversion: {len(self.scheduler.timesteps)} steps x '
              f'{iterations} iterations; attention cached only on the final iteration.', flush=True)
        try:
            for i, t in enumerate(tqdm(self.scheduler.timesteps.flip(0), desc=f'{self.device} fixed-point inversion')):
                # Keep solver arithmetic in fp32, while each UNet evaluation uses
                # the model dtype (fp16 for normal Kaggle inference).
                previous = latent.float()
                def predict(guess, final_iteration):
                    self.cache.begin('invert' if final_iteration else 'off', i)
                    model_input = self.scheduler.scale_model_input(guess.to(clean.dtype), t)
                    return self.unet(model_input, t, embeddings).float()
                latent = fixed_point_solve(previous, predict,
                    lambda noise, anchor: inverse_step(noise, t, anchor, self.scheduler),
                    iterations, relaxation).to(clean.dtype)
                if not torch.isfinite(latent).all():
                    raise RuntimeError('Fixed-point inversion produced nonfinite latents; '
                                       'try a lower inversion_relaxation (e.g. 0.5).')
                self.cache.finish()
        finally:
            self.cache.mode = 'off'
            self.cache.snapshot = {}
        return latent

    @torch.inference_mode()
    def denoise(self, noisy, prompt, source_prompt=None, guidance=7.5, negative=''):
        latent = noisy.clone()
        conditional = self.embed(prompt)
        cfg = guidance > 1
        embeddings = torch.cat([self.embed(negative), conditional]) if cfg else conditional
        if source_prompt is not None:
            source_ids = self.tokenizer(source_prompt, padding='max_length', max_length=77, truncation=True).input_ids
            target_ids = self.tokenizer(prompt, padding='max_length', max_length=77, truncation=True).input_ids
            self.cache.mapper, self.cache.matched = align_tokens(source_ids, target_ids)
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc=f'{self.device} edit' if source_prompt else f'{self.device} reconstruction')):
            self.cache.begin('edit' if source_prompt else 'off', i)
            x = torch.cat([latent, latent]) if cfg else latent
            noise = self.unet(self.scheduler.scale_model_input(x, t), t, embeddings)
            if cfg:
                unconditional, conditional_noise = noise.chunk(2)
                noise = unconditional + guidance * (conditional_noise - unconditional)
            latent = self.scheduler.step(noise, t, latent, eta=0).prev_sample
            self.cache.finish()
        self.cache.mode = 'off'
        return latent

    @torch.inference_mode()
    def decode(self, latent):
        from PIL import Image
        frames = []
        scale = getattr(self.vae.config, 'scaling_factor', .18215)
        for frame in latent[0].permute(1, 0, 2, 3).split(1):
            pixels = (self.vae.decode(frame / scale).sample / 2 + .5).clamp(0, 1)
            array = (pixels[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype('uint8')
            frames.append(Image.fromarray(array))
        return frames
