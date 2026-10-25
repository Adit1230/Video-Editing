"""Reproduce repeated-run disk exhaustion using small files and a simulated disk."""
import ast
import errno
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from cache_manager import CacheManager, atomic_save, cleanup_incomplete, DATA, GiB

OLD = 'a'*20
CURRENT = 'b'*20


def config(**kwargs):
    value=dict(steps=3,keep_cache=True,reuse_cache=True,cache_auto_cleanup=True,
               cache_keep_runs=3,cache_reserve_gib=256/GiB)
    value.update(kwargs)
    return value


def fixture(parent,key=OLD,size=400,done=True):
    root=parent/key;root.mkdir()
    (root/'000.pt').write_bytes(b'x'*size)
    (root/'001.pt').write_bytes(b'x'*size)
    if done:(root/'inversion.pt').write_bytes(b'z'*size)
    return root


def disk(parent,capacity):
    def usage(_):
        size=sum(p.stat().st_size for p in parent.rglob('*') if p.is_file() and DATA.fullmatch(p.name))
        return SimpleNamespace(total=capacity,used=size,free=capacity-size)
    return usage


def snapshot_class(module,name):
    # Execute the actual recorder class without loading Torch/models. Its writer
    # is replaced with a fixed-size serializer; lifecycle/disk code is unchanged.
    path=Path(__file__).parent/module
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    def save(value,path):Path(path).write_bytes(b'x'*256)
    namespace=dict(Path=Path,torch=SimpleNamespace(save=save),shutil=shutil,atomic_save=atomic_save)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


class CacheLifecycleTests(unittest.TestCase):
    def test_reported_first_step_failure_recovers_both_recorders(self):
        for module,name in [('engine.py','AttentionCache'),('attention_ext.py','SourceAttention')]:
            with self.subTest(module=module),tempfile.TemporaryDirectory() as temp:
                parent=Path(temp);old=fixture(parent)
                # 300 bytes initially free. First snapshot leaves 44 bytes,
                # so old behavior fails immediately; cleanup must free old data.
                with patch('cache_manager.shutil.disk_usage',side_effect=disk(parent,1500)):
                    with CacheManager(parent,CURRENT,config()) as manager:
                        cls=snapshot_class(module,name)
                        recorder=cls(manager.path,3) if name=='AttentionCache' else cls(manager.path,3,dict(cache_max_gib=8))
                        recorder.disk_manager=manager
                        for i in range(3):
                            recorder.mode='invert';recorder.step=i;recorder.snapshot={'dummy':1};recorder.finish()
                        self.assertFalse(old.exists())
                        (manager.path/'inversion.pt').write_bytes(b'committed')
                        manager.mark_complete()
                        snapshot=(manager.path/'000.pt').read_bytes()
                    # Same-video / new-target rerun reuses files without resetting.
                    with CacheManager(parent,CURRENT,config()) as manager:
                        self.assertTrue(manager.reusable())
                        self.assertEqual((manager.path/'000.pt').read_bytes(),snapshot)

    def test_first_snapshot_is_not_counted_twice(self):
        for module,name in [('engine.py','AttentionCache'),('attention_ext.py','SourceAttention')]:
            with self.subTest(module=module),tempfile.TemporaryDirectory() as temp:
                cls=snapshot_class(module,name)
                recorder=cls(temp,3) if name=='AttentionCache' else cls(temp,3,dict(cache_max_gib=8))
                recorder.mode='invert';recorder.snapshot={}
                with patch('shutil.disk_usage',return_value=SimpleNamespace(free=2*GiB+512)):
                    recorder.finish()  # remaining 512 + reserve fits exactly

    def test_failed_inversion_is_removed_but_completed_inversion_survives_edit_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp)
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                with CacheManager(parent,CURRENT,config(cache_reserve_gib=0)) as manager:
                    (manager.path/'000.pt').write_bytes(b'partial')
                    raise RuntimeError('interrupted')
            self.assertFalse((parent/CURRENT).exists())
            with CacheManager(parent,CURRENT,config(steps=1,cache_reserve_gib=0)) as manager:
                (manager.path/'000.pt').write_bytes(b'valid')
                (manager.path/'inversion.pt').write_bytes(b'valid')
                manager.mark_complete()
            with self.assertRaisesRegex(RuntimeError,'edit failed'):
                with CacheManager(parent,CURRENT,config(steps=1,cache_reserve_gib=0)) as manager:
                    raise RuntimeError('edit failed')
            self.assertTrue((parent/CURRENT/'inversion.pt').is_file())

    def test_stale_partial_files_cleaned_unrelated_files_and_symlinks_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp);partial=fixture(parent,done=False)
            unknown=parent/('c'*20);unknown.mkdir();(unknown/'my-video.mp4').write_bytes(b'user')
            link=parent/('d'*20);link.symlink_to(unknown,target_is_directory=True)
            with CacheManager(parent,CURRENT,config(cache_reserve_gib=0)) as manager:
                self.assertFalse(partial.exists())
                self.assertTrue((unknown/'my-video.mp4').is_file());self.assertTrue(link.is_symlink())

    def test_active_cache_and_matching_cache_are_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp)
            with CacheManager(parent,OLD,config(steps=1,cache_reserve_gib=0)) as active:
                (active.path/'000.pt').write_bytes(b'active')
                (active.path/'inversion.pt').write_bytes(b'active');active.mark_complete()
                with self.assertRaisesRegex(RuntimeError,'another process'):
                    with CacheManager(parent,OLD,config(steps=1,cache_reserve_gib=0)):pass
                with patch('cache_manager.shutil.disk_usage',side_effect=disk(parent,20)):
                    with CacheManager(parent,CURRENT,config(steps=1,cache_reserve_gib=0)) as other:
                        with self.assertRaisesRegex(RuntimeError,'Insufficient attention-cache disk'):
                            other.ensure_space(100)
                        self.assertTrue((active.path/'000.pt').is_file())

    def test_preflight_cleans_only_interrupted_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp);old=fixture(parent)
            interrupted=fixture(parent,CURRENT,done=False)
            with CacheManager(parent,'c'*20,config(steps=1,cache_reserve_gib=0)) as active:
                (active.path/'000.pt').write_bytes(b'working')
                cleanup_incomplete(parent)
                self.assertFalse(interrupted.exists())
                self.assertTrue(old.exists())
                self.assertTrue((active.path/'000.pt').is_file())

    def test_retention_and_auto_cleanup_can_be_configured(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp);old=fixture(parent)
            with CacheManager(parent,CURRENT,config(cache_auto_cleanup=False,cache_reserve_gib=0)) as manager:
                self.assertTrue(old.exists())
                with patch('cache_manager.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
                    with self.assertRaises(RuntimeError):manager.ensure_space(1)
                self.assertTrue(old.exists())
            with CacheManager(parent,CURRENT,config(cache_keep_runs=1,cache_reserve_gib=0)):
                self.assertFalse(old.exists())

    def test_atomic_write_failure_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'000.pt';path.write_bytes(b'old-complete')
            def failed(value,temporary):
                temporary.write_bytes(b'partial')
                raise OSError(errno.ENOSPC,'disk full')
            with self.assertRaises(OSError):atomic_save({},path,failed)
            self.assertEqual(path.read_bytes(),b'old-complete')
            self.assertFalse(path.with_name('000.pt.tmp').exists())

    def test_incomplete_current_and_keep_false(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp);current=fixture(parent,CURRENT)
            with CacheManager(parent,CURRENT,config(steps=3,cache_reserve_gib=0)) as manager:
                self.assertFalse((current/'inversion.pt').exists())  # only 2/3 snapshots
                for i in range(3):(current/f'{i:03d}.pt').write_bytes(b'ok')
                (current/'inversion.pt').write_bytes(b'ok');manager.mark_complete()
            with CacheManager(parent,CURRENT,config(keep_cache=False,cache_reserve_gib=0)):
                pass
            self.assertFalse(current.exists())

    def test_stream_spill_interruption_preserves_reusable_noisy_latent(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp)
            with self.assertRaisesRegex(RuntimeError,'edit failed'):
                with CacheManager(parent,CURRENT,config(),attention=False,latents=False) as manager:
                    (manager.path/'inversion.pt').write_bytes(b'noisy')
                    manager.mark_complete()
                    (manager.path/f'stream_{OLD}.pt').write_bytes(b'current-step')
                    raise RuntimeError('edit failed')
            with CacheManager(parent,CURRENT,config(),attention=False,latents=False) as manager:
                self.assertTrue(manager.reusable())
                self.assertEqual((manager.path/'inversion.pt').read_bytes(),b'noisy')


if __name__=='__main__':unittest.main()
