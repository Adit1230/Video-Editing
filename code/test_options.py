"""No-model regression checks for configuration and dispatch."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from options import normalized_config,validate_options,modern_runtime,MODELS


class OptionTests(unittest.TestCase):
    def config(self,**kw):
        c=json.loads((Path(__file__).parent/'config.example.json').read_text());c.update(kw)
        return normalized_config(c)

    def test_all_backends_and_masks(self):
        for backend in MODELS:
            for mask in ('attention','sam2','grounded_sam2','hybrid','file'):
                c=self.config(backend=backend,model_id=MODELS[backend][0],
                    frames=9 if backend=='cogvideox_2b' else 4,mask_backend=mask,
                    mask_dir='/tmp/masks',mask_boxes=[[0,0,20,20]],mask_prompt='flowers')
                validate_options(c)
                self.assertEqual(modern_runtime(c),backend!='sd14' or mask!='attention')

    def test_invalid_combinations(self):
        variants=[dict(backend='cogvideox_2b',frames=4),dict(mask_backend='sam2',mask_boxes=[],mask_points=[]),
            dict(mask_backend='hybrid',mask_hybrid_source='file',mask_dir=''),
            dict(mask_backend='file',mask_dir=''),dict(runtime='legacy',backend='sd21'),
            dict(inversion_method='direct',inversion_relaxation=.5),dict(attention_storage='unknown'),
            dict(attention_heads=-1),dict(attention_heads=1.5),dict(attention_max_queries=0),
            dict(attention_storage='stream',stream_cache_ram_gib=-1),
            dict(attention_storage='disk',backend='cogvideox_2b',frames=9,attention_max_queries=0)]
        for kw in variants:
            with self.subTest(kw=kw),self.assertRaises(ValueError):validate_options(self.config(**kw))

    def test_legacy_direct_dispatch(self):
        import launch
        c=self.config(inversion_method='direct',inversion_iterations=5)
        with patch('run.main') as execute:
            launch.main(c)
            sent=execute.call_args.args[0]
            self.assertEqual(sent['inversion_iterations'],1)
            self.assertEqual(sent['inversion_method'],'fixed_point')

    def test_stream_is_independent_of_model_and_solver(self):
        for backend in MODELS:
            for method in ('direct','fixed_point'):
                c=self.config(backend=backend,model_id=MODELS[backend][0],attention_storage='stream',
                    attention_heads=0,attention_max_queries=0,
                    frames=9 if backend=='cogvideox_2b' else 4,inversion_method=method)
                validate_options(c)
                self.assertTrue(modern_runtime(c))
        self.assertEqual(normalized_config(dict(backend='sd21',attention_storage='stream'))['attention_heads'],0)


if __name__=='__main__':unittest.main()
