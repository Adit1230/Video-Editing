"""Source-video segmentation stage. Models are released before diffusion starts."""
from pathlib import Path
import gc
import hashlib
import json
import os
import tempfile
import numpy as np
from PIL import Image,ImageOps,ImageFilter

SAM_COMMIT='2b90b9f5ceec907a1c18123530e92e794ad901a4'


def transform_mask(mask,size,dilation=0,feather=0):
    mask=ImageOps.fit(mask.convert('L'),(size,size),method=Image.Resampling.NEAREST)
    if dilation:mask=mask.filter(ImageFilter.MaxFilter(2*int(dilation)+1))
    if feather:mask=mask.filter(ImageFilter.GaussianBlur(float(feather)))
    return mask


def frame_stream(c,indices):
    """Yield global source indices and untransformed RGB frames, for dense tracking."""
    from run import resolve_video
    path=resolve_video(c)
    wanted=set(indices)
    if path.is_dir():
        files=sorted(p for p in path.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp'))
        for i in indices:
            with Image.open(files[i]) as im:yield i,im.convert('RGB')
    else:
        import cv2
        cap=cv2.VideoCapture(str(path))
        try:
            for i in range(max(indices)+1):
                ok,frame=cap.read()
                if not ok:raise ValueError(f'Video ended before mask-tracking frame {i}')
                if i in wanted:yield i,Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        finally:cap.release()


def grounding_boxes(image,c,device):
    import torch
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    model_id=c['grounding_model_id']
    processor=AutoProcessor.from_pretrained(model_id,revision=c['resolved_grounding_revision'])
    model=AutoModelForZeroShotObjectDetection.from_pretrained(model_id,revision=c['resolved_grounding_revision']).to(device).eval()
    prompt=c['mask_prompt'].strip().rstrip('.')+'.'
    inputs=processor(images=image,text=prompt,return_tensors='pt').to(device)
    with torch.inference_mode():outputs=model(**inputs)
    results=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,
        threshold=c.get('grounding_box_threshold',.3),text_threshold=c.get('grounding_text_threshold',.25),
        target_sizes=[image.size[::-1]])[0]
    order=results['scores'].argsort(descending=True)
    boxes=results['boxes'][order].float().cpu().numpy()
    del model,processor,outputs,inputs
    gc.collect();torch.cuda.empty_cache()
    if not len(boxes):raise RuntimeError('No objects detected for mask_prompt. Adjust it or supply points/boxes.')
    if c['mask_instance']!='all':
        instance=int(c['mask_instance'])
        if not 0<=instance<len(boxes):raise ValueError(f'mask_instance must be <{len(boxes)}')
        boxes=boxes[instance:instance+1]
    return boxes


def sam_track(c,track,selected,destination,device,*,predictor_factory=None):
    """Track masks; a supplied factory allows contract tests without optional SAM 2."""
    import torch
    if predictor_factory is None:
        try:from sam2.build_sam import build_sam2_video_predictor
        except ImportError as e:raise RuntimeError('Install requirements-segmentation.txt in the modern runtime (not the notebook kernel).') from e
        predictor_factory=build_sam2_video_predictor
    from huggingface_hub import hf_hub_download
    configs={
        'facebook/sam2.1-hiera-tiny':('t','tiny'),
        'facebook/sam2.1-hiera-small':('s','small'),
        'facebook/sam2.1-hiera-base-plus':('b+','base_plus'),
        'facebook/sam2.1-hiera-large':('l','large'),
    }
    if c['sam_model_id'] not in configs:raise ValueError('Use an official facebook/sam2.1-hiera-{tiny,small,base-plus,large} checkpoint')
    code,name=configs[c['sam_model_id']]
    checkpoint=hf_hub_download(c['sam_model_id'],f'sam2.1_hiera_{name}.pt',revision=c['resolved_sam_revision'],token=os.environ.get('HF_TOKEN'))
    predictor=predictor_factory(f'configs/sam2.1/sam2.1_hiera_{code}.yaml',checkpoint,device=device,apply_postprocessing=False)
    with tempfile.TemporaryDirectory(prefix='video-editing-track-') as tmp:
        folder=Path(tmp)
        first=None
        for position,(index,image) in enumerate(frame_stream(c,track)):
            if first is None:first=image.copy()
            image.save(folder/f'{position:06d}.jpg',quality=95)
        boxes=np.asarray(c['mask_boxes'],dtype=np.float32)
        source=c['mask_hybrid_source'] if c['mask_backend']=='hybrid' else c['mask_backend']
        if source=='grounded_sam2':boxes=grounding_boxes(first,c,device)
        points=np.asarray(c['mask_points'],dtype=np.float32) if c['mask_points'] else None
        labels=np.asarray(c['mask_labels'],dtype=np.int32) if c['mask_labels'] else None
        selected_positions={track.index(index):i for i,index in enumerate(selected)}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            state=predictor.init_state(video_path=str(folder),offload_video_to_cpu=True,offload_state_to_cpu=True)
            if len(boxes):
                for object_id,box in enumerate(boxes,1):
                    kwargs=dict(inference_state=state,frame_idx=0,obj_id=object_id,box=box)
                    if object_id==1 and points is not None:kwargs.update(points=points,labels=labels)
                    predictor.add_new_points_or_box(**kwargs)
            elif points is not None:
                predictor.add_new_points_or_box(state,0,1,points=points,labels=labels)
            else:raise ValueError('SAM2 needs boxes or points')
            for correction in c.get('mask_corrections',[]):
                original=correction['frame_index']
                if original not in track:raise ValueError('Correction frame must be in the tracked frame indices')
                kwargs=dict(inference_state=state,frame_idx=track.index(original),obj_id=correction.get('object_id',1))
                for key in ('points','labels','box'):
                    if key in correction:kwargs[key]=np.asarray(correction[key])
                predictor.add_new_points_or_box(**kwargs)
            for position,ids,logits in predictor.propagate_in_video(state):
                if position in selected_positions:
                    mask=(logits>0).any(0).squeeze(0).cpu().numpy().astype('uint8')*255
                    Image.fromarray(mask).save(destination/f'{selected_positions[position]:05d}.png')
        predictor.reset_state(state)
        del state
    del predictor
    gc.collect();torch.cuda.empty_cache()


def build_masks(c,selected,source_images,out,device='cuda:0'):
    """Return CPU [frames,1,H,W] masks. None retains original attention masking."""
    import torch
    if c['mask_backend']=='attention':return None,None
    source=c['mask_hybrid_source'] if c['mask_backend']=='hybrid' else c['mask_backend']
    stride=c['mask_track_stride']
    for correction in c.get('mask_corrections',[]):
        if not selected[0]<=correction['frame_index']<=selected[-1]:
            raise ValueError('Correction frame must lie within the selected clip interval')
    track=sorted(set(range(selected[0],selected[-1]+1,stride))|set(selected)|{item['frame_index'] for item in c.get('mask_corrections',[])})
    if source!='file' and len(track)>c.get('mask_max_track_frames',300):
        raise ValueError('Too many dense mask-tracking frames. Increase mask_track_stride or mask_max_track_frames.')
    from huggingface_hub import model_info
    if source!='file':
        c['resolved_sam_revision']=model_info(c['sam_model_id'],revision=c['sam_revision']).sha
    if source=='grounded_sam2':
        c['resolved_grounding_revision']=model_info(c['grounding_model_id'],revision=c['grounding_revision']).sha
    keys=('mask_backend','mask_hybrid_source','mask_prompt','sam_model_id','resolved_sam_revision',
          'grounding_model_id','resolved_grounding_revision','mask_instance','mask_boxes','mask_points','mask_labels',
          'mask_corrections','grounding_box_threshold','grounding_text_threshold','size','mask_track_stride')
    h=hashlib.sha256(json.dumps({k:c.get(k) for k in keys},sort_keys=True).encode())
    h.update(Path(__file__).read_bytes())
    # Hash tracking images too: unseen in-between motion affects propagation.
    for i,image in frame_stream(c,selected if source=='file' else track):
        h.update(str((i,image.size)).encode());h.update(image.tobytes())
    files=[]
    if source=='file':
        for i in selected:
            matches=[Path(c['mask_dir'])/f'{i:05d}.png',Path(c['mask_dir'])/f'{i:06d}.png',Path(c['mask_dir'])/f'{i}.png']
            file=next((p for p in matches if p.is_file()),None)
            if file is None:raise FileNotFoundError(f'Missing mask for SOURCE frame {i} in {c["mask_dir"]}')
            files.append(file);h.update(file.read_bytes())
    key=h.hexdigest()[:20]
    cached=Path(c['mask_cache_path'])/key
    cached.mkdir(parents=True,exist_ok=True)
    required=[cached/f'{i:05d}.png' for i in range(len(selected))]
    if not (cached/'complete.json').exists() or not all(p.is_file() for p in required):
        if source=='file':
            for file,dest in zip(files,required):
                with Image.open(file) as im:im.convert('L').save(dest)
        else:sam_track(c,track,selected,cached,device)
        if not all(p.is_file() for p in required):raise RuntimeError('Segmentation did not produce every selected frame')
        (cached/'complete.json').write_text(json.dumps(dict(selected=selected,track=track,mask_source=source,sam_code_commit=SAM_COMMIT)))
    masks=[];previews=[]
    preview_dir=Path(out)/'mask_frames';preview_dir.mkdir(exist_ok=True)
    for i,(file,image) in enumerate(zip(required,source_images)):
        with Image.open(file) as raw:
            mask=transform_mask(raw,c['size'],c['mask_dilation'],c['mask_feather'])
        array=np.asarray(mask).astype('float32')/255
        if array.max()==0 and not c['mask_allow_empty']:
            raise RuntimeError(f'Empty mask at selected frame {i}. Correct the mask or set mask_allow_empty=True for a known occlusion.')
        mask.save(preview_dir/f'{i:05d}.png')
        masks.append(torch.from_numpy(array.copy())[None])
        rgb=np.asarray(image).astype('float32')
        overlay=(rgb*(1-.35*array[...,None])+np.array([255,40,40])*.35*array[...,None]).clip(0,255).astype('uint8')
        previews.append(Image.fromarray(overlay))
    from run import export_video
    export_video(previews,Path(out)/'mask_preview.mp4',c['output_fps'])
    previews[0].save(Path(out)/'mask_preview_first_frame.png')
    print('Saved mask_preview.mp4; mask cache:',key,flush=True)
    return torch.stack(masks),key
