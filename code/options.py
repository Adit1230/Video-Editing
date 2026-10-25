"""Configuration shared by the notebook and both runtimes (no torch import)."""
from copy import deepcopy
import math
from inversion import validate_fixed_point

MODELS = {
    'sd14': ('CompVis/stable-diffusion-v1-4', '2880f2ca379f41b0226444936bb7a6766a227587'),
    'sd21': ('sd2-community/stable-diffusion-2-1-base', '4e63672c03103b6c636b8fb4119ba982469b2955'),
    'cogvideox_2b': ('THUDM/CogVideoX-2b', '1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01'),
}
EDIT_PROFILES = {
    'original': dict(self_replace=.5, cross_replace=.3, sparse_least_channel=640,
        zero_temporal_bias=False, negative_prompt_inversion=False,
        mask_latent_blend=False, blend_fraction=1., blend_dilate=0, blend_feather=0),
    'improved': dict(self_replace=.7, cross_replace=.5, sparse_least_channel=320,
        zero_temporal_bias=True, negative_prompt_inversion=True,
        mask_latent_blend=True, blend_fraction=1., blend_dilate=1, blend_feather=1),
}

MASK_MODES = ('attention', 'sam2', 'grounded_sam2', 'hybrid', 'file')


def normalized_config(config):
    c = deepcopy(config)
    c.setdefault('edit_profile', 'original')
    if c['edit_profile'] not in EDIT_PROFILES:raise ValueError('edit_profile must be original or improved')
    for name,value in EDIT_PROFILES[c['edit_profile']].items():c.setdefault(name,value)
    # Accept the friend's spelling; mask_latent_blend remains the canonical key.
    if 'latent_blend' in c:c['mask_latent_blend']=c.pop('latent_blend')
    c.setdefault('negative_prompt', '')
    c.setdefault('backend', 'sd14')
    c.setdefault('mask_backend', 'attention')
    c.setdefault('inversion_method', 'fixed_point')
    c.setdefault('inversion_iterations', 2)
    c.setdefault('inversion_relaxation', 1.)
    c.setdefault('runtime', 'auto')
    c.setdefault('model_cpu_offload', True)
    c.setdefault('vae_fp32', c['backend']=='cogvideox_2b')
    c.setdefault('cache_attention', True)
    c.setdefault('attention_storage', 'disk')
    c.setdefault('stream_cache_ram_gib', 0.)
    c.setdefault('stream_cache_device', 'auto')
    c.setdefault('stream_cache_gpu_gib', 0.)
    c.setdefault('stream_gpu_reserve_gib', 4.)
    c.setdefault('stream_profile', False)
    c.setdefault('attention_layers', [0, 10, 20])
    c.setdefault('attention_heads', 0 if c['attention_storage']=='stream' else 2)
    c.setdefault('attention_query_chunk', 128)
    c.setdefault('attention_max_queries', 1024)
    c.setdefault('cache_max_gib', 8.)
    c.setdefault('cache_auto_cleanup', True)
    c.setdefault('cache_keep_runs', 2)
    c.setdefault('cache_reserve_gib', 2.)
    c.setdefault('mask_prompt', ' '.join(c.get('edit_words', [])))
    c.setdefault('sam_model_id', 'facebook/sam2.1-hiera-tiny')
    c.setdefault('sam_revision', 'de431c4043854a71d8101e17995dfe596bf101a5')
    c.setdefault('grounding_model_id', 'IDEA-Research/grounding-dino-tiny')
    c.setdefault('grounding_revision', 'a2bb814dd30d776dcf7e30523b00659f4f141c71')
    c.setdefault('mask_instance', 'all')
    c.setdefault('mask_boxes', [])
    c.setdefault('mask_points', [])
    c.setdefault('mask_labels', [])
    c.setdefault('mask_dir', '')
    c.setdefault('mask_dilation', 4)
    c.setdefault('mask_feather', 3)
    c.setdefault('mask_threshold', .3)
    c.setdefault('mask_hybrid_source', 'grounded_sam2')
    c.setdefault('mask_cache_path', '/kaggle/working/video_editing_mask_cache')
    c.setdefault('mask_track_stride', 1)
    c.setdefault('mask_allow_empty', False)
    c.setdefault('mask_corrections', [])
    c.setdefault('mask_only', False)
    c.setdefault('mask_latent_blend', False)
    c.setdefault('mask_composite', False)
    c.setdefault('edit_words', [])
    if c['backend'] not in MODELS:
        raise ValueError(f'backend must be one of {list(MODELS)}')
    default_id, default_revision = MODELS[c['backend']]
    if not c.get('model_id'):
        c['model_id'], c['model_revision'] = default_id, default_revision
    c.setdefault('model_revision', default_revision)
    return c


def modern_runtime(c):
    return c.get('runtime') == 'modern' or c.get('attention_storage','disk')=='stream' or c['backend'] != 'sd14' or c['mask_backend'] != 'attention' or c.get('mask_latent_blend') or not c.get('cache_attention', True) or c.get('zero_temporal_bias') or c.get('sparse_least_channel',640)!=640 or c.get('negative_prompt_inversion') or c.get('compute_metrics',False)


def validate_options(c):
    for name in ('zero_temporal_bias','negative_prompt_inversion','mask_latent_blend'):
        if not isinstance(c[name],bool):raise ValueError(f'{name} must be True or False')
    if isinstance(c['sparse_least_channel'],bool) or c['sparse_least_channel'] not in (0,320,640,1280):
        raise ValueError('sparse_least_channel must be 0, 320, 640 or 1280')
    if isinstance(c['blend_fraction'],bool) or not isinstance(c['blend_fraction'],(int,float)) or not math.isfinite(c['blend_fraction']) or not 0<=c['blend_fraction']<=1:
        raise ValueError('blend_fraction must be finite in [0,1]')
    for name in ('blend_dilate','blend_feather'):
        if isinstance(c[name],bool) or not isinstance(c[name],int) or not 0<=c[name]<=32:
            raise ValueError(f'{name} must be an integer in [0,32]')
    if not isinstance(c['negative_prompt'],str):raise ValueError('negative_prompt must be a string')
    if c['mask_latent_blend'] and c['mask_backend']=='attention' and not c.get('edit_words'):
        raise ValueError('Attention latent blending needs edit_words; alternatively select segmentation masks')
    if not isinstance(c['cache_auto_cleanup'],bool):
        raise ValueError('cache_auto_cleanup must be True or False')
    if isinstance(c['cache_keep_runs'],bool) or not isinstance(c['cache_keep_runs'],int) or c['cache_keep_runs']<1:
        raise ValueError('cache_keep_runs must be an integer >=1')
    if not isinstance(c['cache_reserve_gib'],(float,int)) or not math.isfinite(c['cache_reserve_gib']) or c['cache_reserve_gib']<0:
        raise ValueError('cache_reserve_gib must be finite and >=0')
    if c['mask_backend'] not in MASK_MODES:
        raise ValueError(f'mask_backend must be one of {MASK_MODES}')
    if c['runtime'] not in ('auto', 'legacy', 'modern'):
        raise ValueError('runtime must be auto, legacy or modern')
    if c['runtime'] == 'legacy' and modern_runtime(dict(c, runtime='auto')):
        raise ValueError('This configuration needs the modern runtime. Use runtime=auto.')
    if c['attention_storage'] not in ('disk','stream'):
        raise ValueError('attention_storage must be disk or stream')
    if c['stream_cache_device'] not in ('auto','cpu'):raise ValueError('stream_cache_device must be auto or cpu')
    if not isinstance(c['stream_profile'],bool):raise ValueError('stream_profile must be True or False')
    for key in ('stream_cache_gpu_gib','stream_gpu_reserve_gib'):
        if isinstance(c[key],bool) or not isinstance(c[key],(int,float)) or not math.isfinite(c[key]) or c[key]<0:
            raise ValueError(f'{key} must be finite and >=0')
    if not isinstance(c['stream_cache_ram_gib'],(int,float)) or not math.isfinite(c['stream_cache_ram_gib']) or c['stream_cache_ram_gib']<0:
        raise ValueError('stream_cache_ram_gib must be finite and >=0 (0 selects available RAM automatically)')
    if c['inversion_method'] not in ('direct', 'fixed_point'):
        raise ValueError('inversion_method must be direct or fixed_point')
    validate_fixed_point(c['inversion_iterations'], c['inversion_relaxation'])
    if c['inversion_method'] == 'direct' and c['inversion_relaxation'] != 1:
        raise ValueError('Direct DDIM inversion requires inversion_relaxation=1')
    if not c.get('source_prompt', '').strip() or not c.get('target_prompt', '').strip():
        raise ValueError('Set source_prompt and target_prompt')
    if isinstance(c.get('guidance',7.5),bool) or not isinstance(c.get('guidance',7.5),(int,float)) or not math.isfinite(c.get('guidance',7.5)) or c.get('guidance', 7.5) <= 1:
        raise ValueError('Editing guidance must be >1')
    for key in ('self_replace', 'cross_replace'):
        if not 0 <= c.get(key, .5) <= 1:
            raise ValueError(f'{key} must be in [0,1]')
    if isinstance(c['steps'],bool) or not isinstance(c['steps'], int) or not 2 <= c['steps'] <= 100:
        raise ValueError('steps must be an integer in [2,100]')
    if isinstance(c['frames'],bool) or not isinstance(c['frames'],int):
        raise ValueError('frames must be an integer')
    if c.get('clip_seconds') not in (None,'auto'):
        seconds=c['clip_seconds']
        if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0:
            raise ValueError('clip_seconds must be positive, "auto", or null')
    if c.get('start_frame',0)<0 or c.get('frame_stride',1)<1:
        raise ValueError('start_frame must be >=0 and frame_stride >=1')
    if c['frames'] < 2 or c['frames'] > 49:
        raise ValueError('Use 2-49 frames; the image backends support at most 16')
    if isinstance(c['size'],bool) or not isinstance(c['size'],int):
        raise ValueError('size must be an integer')
    if c['size'] % 16 or c['size'] < 128:
        raise ValueError('size must be a multiple of 16 and >=128')
    if c['backend'] != 'cogvideox_2b' and c['frames'] > 16:
        raise ValueError('Image backends support at most 16 frames')
    if c['backend'] == 'cogvideox_2b':
        if (c['frames'] - 1) % 4:
            raise ValueError('CogVideoX frame count must be 4*k+1 (e.g. 9,17,25,49)')
        if not c['attention_layers'] or any(not isinstance(i, int) or i < 0 for i in c['attention_layers']):
            raise ValueError('Select nonnegative attention_layers for the native model')
    for name in ('attention_heads','attention_query_chunk','attention_max_queries'):
        if isinstance(c[name],bool) or not isinstance(c[name],int) or c[name]<(1 if name=='attention_query_chunk' else 0):
            raise ValueError('Attention sizes must be positive integers; attention_heads=0 selects all heads')
    if c['attention_max_queries']==0 and c['attention_storage']!='stream':
        raise ValueError('attention_max_queries=0 auto-fits full query grids in stream mode; choose a positive disk bound')
    if not isinstance(c['cache_max_gib'],(float,int)) or not math.isfinite(c['cache_max_gib']) or c['cache_max_gib']<=0:
        raise ValueError('cache_max_gib must be finite and positive (disk inversion storage only)')
    if c['mask_dilation'] < 0 or c['mask_feather'] < 0 or c['mask_track_stride'] < 1:
        raise ValueError('Mask dilation/feather must be >=0 and track stride >=1')
    if c['mask_backend'] == 'file' and not c['mask_dir']:
        raise ValueError('Set mask_dir to masks named by source frame index')
    source = c['mask_hybrid_source'] if c['mask_backend'] == 'hybrid' else c['mask_backend']
    if source not in MASK_MODES or source == 'hybrid':
        raise ValueError('mask_hybrid_source must be sam2, grounded_sam2 or file')
    if source == 'file' and not c['mask_dir']:
        raise ValueError('Set mask_dir for external file masks')
    if source == 'sam2' and not c['mask_boxes'] and not c['mask_points']:
        raise ValueError('sam2 needs mask_boxes or mask_points. For text-only masks use grounded_sam2.')
    if any(len(box)!=4 or box[2]<=box[0] or box[3]<=box[1] for box in c['mask_boxes']):
        raise ValueError('mask_boxes must contain [x1,y1,x2,y2] with positive width/height')
    if any(len(point)!=2 for point in c['mask_points']):
        raise ValueError('mask_points must contain [x,y] coordinates')
    if any(label not in (0,1) for label in c['mask_labels']):
        raise ValueError('mask_labels must be 0 or 1')
    if len(c['mask_points']) != len(c['mask_labels']):
        raise ValueError('mask_points and mask_labels must have equal lengths')
    if source == 'grounded_sam2' and not c['mask_prompt'].strip():
        raise ValueError('grounded_sam2 needs mask_prompt')
    if not c['cache_attention'] and c['mask_backend']=='attention' and c['mask_latent_blend']:
        raise ValueError('Attention-derived latent masks need cache_attention=True')
    if c.get('mask_composite') and c['mask_backend'] == 'attention':
        raise ValueError('mask_composite requires cached external masks')
    if c['mask_backend']=='hybrid' and c['mask_hybrid_source'] not in ('sam2','grounded_sam2','file'):
        raise ValueError('Hybrid source must be sam2, grounded_sam2 or file')
    if c['backend'] == 'sd14' and c['model_id'] == MODELS['sd21'][0]:
        raise ValueError('Select backend=sd21 for an SD2.1 checkpoint')
    return c
