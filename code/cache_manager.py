"""Disk lifecycle for regenerable inversion caches; no model dependencies.

Only known pipeline cache files in 20-hex fingerprint directories are eligible
for eviction. Model downloads, masks, exports and unrelated directories are not.
"""
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

KEY = re.compile(r'[0-9a-f]{20}\Z')
DATA = re.compile(r'(?:\d{3}|latent_\d{3}|inversion|stream_[0-9a-f]{20})\.pt(?:\.tmp)?\Z')
CONTROL = {'cache.lock', '_cache.json', '_cache.json.tmp'}
GiB = 2**30


def atomic_save(value, path, save):
    """Save beside the destination and rename only a complete file into place."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def managed(path):
    path = Path(path)
    if path.is_symlink() or not path.is_dir() or not KEY.fullmatch(path.name):
        return False
    return all(not p.is_symlink() and p.is_file() and
               (p.name in CONTROL or DATA.fullmatch(p.name)) for p in path.iterdir())


def metadata(path):
    try:
        value = json.loads((Path(path) / '_cache.json').read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def complete(path, steps=None, attention=None, latents=None):
    path = Path(path)
    state = path / 'inversion.pt'
    if not state.is_file() or state.stat().st_size == 0:
        return False
    info = metadata(path)
    if info.get('status') == 'inverting':
        return False
    n = steps if steps is not None else info.get('steps')
    if n is not None:
        required = []
        needs_attention = attention if attention is not None else info.get('attention', True)
        needs_latents = latents if latents is not None else info.get('latents', False)
        if needs_attention:
            required += [path / f'{i:03d}.pt' for i in range(n)]
        if needs_latents:
            required += [path / f'latent_{i:03d}.pt' for i in range(n)]
        return all(p.is_file() and p.stat().st_size > 0 for p in required)
    # Pre-manager caches commit inversion.pt only after successful inversion.
    return True


def used_bytes(path):
    return sum(p.stat().st_size for p in Path(path).iterdir() if p.is_file() and not p.is_symlink())


class CacheManager:
    def __init__(self, parent, key, config, *, attention=True, latents=False):
        if not KEY.fullmatch(key):
            raise ValueError('Cache key must be a 20-character hex fingerprint')
        self.parent = Path(parent).resolve()
        self.path = self.parent / key
        self.c = config
        self.attention, self.latents = attention, latents
        self.steps = config['steps']
        self.auto = config.get('cache_auto_cleanup', True)
        self.keep_runs = config.get('cache_keep_runs', 2)
        self.reserve = math.ceil(config.get('cache_reserve_gib', 2.) * GiB)
        self.keep = config.get('keep_cache', False)
        self.lock = None
        self.removed = 0
        self.discard = False

    def __enter__(self):
        self.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError('Refusing a symlinked inversion-cache directory')
        self.path.mkdir(exist_ok=True)
        if not managed(self.path):
            raise ValueError(f'Unrecognized files in {self.path}; no files were removed')
        self.lock = (self.path / 'cache.lock').open('a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close(); self.lock = None
            raise RuntimeError('This inversion cache is active in another process. Wait for that run to finish.')
        try:
            if not self.c.get('reuse_cache', True) or not self.reusable():
                self.reset()
            else:
                self._write_info('complete')  # update recency without changing tensors
            if self.auto:
                self.cleanup()
            self.ensure_space(0)
            return self
        except BaseException:
            self.__exit__(*__import__('sys').exc_info())
            raise

    def _write_info(self, status):
        value = dict(schema=1, status=status, steps=self.steps, attention=self.attention,
                     latents=self.latents, last_used=time.time())
        atomic_save(value, self.path / '_cache.json',
                    lambda obj, path: path.write_text(json.dumps(obj)))

    def reusable(self):
        return complete(self.path, self.steps, self.attention, self.latents)

    def reset(self):
        # Current cache is exclusively locked and contains only recognized files.
        if not managed(self.path):
            raise ValueError('Refusing to reset a cache containing unrecognized files')
        for p in self.path.iterdir():
            if p.name != 'cache.lock':
                p.unlink()
        self._write_info('inverting')

    def mark_complete(self):
        # Metadata is written last. A killed or interrupted inversion stays incomplete.
        if not complete_ignoring_status(self.path, self.steps, self.attention, self.latents):
            raise RuntimeError('Inversion did not produce every expected cache file')
        self._write_info('complete')

    def _candidates(self):
        candidates = []
        for path in self.parent.iterdir():
            if path == self.path or not managed(path):
                continue
            info = metadata(path)
            recent = info.get('last_used', path.stat().st_mtime)
            candidates.append((complete(path), float(recent), path))
        return sorted(candidates, key=lambda item: (item[0], item[1]))

    def _evict(self, path, reason):
        removed = remove_inactive(path, reason)
        if removed:
            self.removed += 1
        return removed

    def cleanup(self):
        candidates = self._candidates()
        for done, _, path in candidates:
            if not done:
                self._evict(path, 'incomplete')
        completed = [(recent, path) for done, recent, path in candidates if done and path.exists()]
        # keep_runs counts the protected current cache plus retained older caches.
        remaining = len(completed)
        for _, path in completed:
            if remaining <= max(0, self.keep_runs - 1):
                break
            if self._evict(path, 'older'):
                remaining -= 1

    def ensure_space(self, remaining_bytes):
        needed = max(0, math.ceil(remaining_bytes)) + self.reserve
        free = shutil.disk_usage(self.path).free
        if free < needed and self.auto:
            for done, _, path in self._candidates():
                self._evict(path, 'older' if done else 'incomplete')
                free = shutil.disk_usage(self.path).free
                if free >= needed:
                    break
        if free < needed:
            raise RuntimeError(
                f'Insufficient attention-cache disk: {needed / GiB:.2f} GiB needed '
                f'for remaining data + reserve, {free / GiB:.2f} GiB available '
                f'after cache cleanup. Partial inversion files will be removed. '
                'Free other files or reduce frames/resolution/steps; resetting the environment is unnecessary.')

    def release(self):
        self.discard = True

    def __exit__(self, exc_type, exc, traceback):
        if self.lock is None:
            return False
        try:
            if self.discard or not self.keep or not self.reusable():
                if managed(self.path):
                    shutil.rmtree(self.path)
            elif self.path.is_dir():
                # A denoising/export failure retains a fully committed source inversion.
                self._write_info('complete')
        finally:
            self.lock.close(); self.lock = None
        return False


def complete_ignoring_status(path, steps, attention, latents):
    path = Path(path)
    names = ['inversion.pt']
    if attention:
        names += [f'{i:03d}.pt' for i in range(steps)]
    if latents:
        names += [f'latent_{i:03d}.pt' for i in range(steps)]
    return all((path/name).is_file() and (path/name).stat().st_size > 0 for name in names)


def remove_inactive(path, reason):
    # Opening a separate lock descriptor also detects active caches in this process.
    path = Path(path)
    try:
        handle = (path / 'cache.lock').open('a+')
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if not managed(path):
            return False
        size = used_bytes(path)
        shutil.rmtree(path)
        print(f'Cache cleanup: removed {reason} inversion cache ({size / GiB:.2f} GiB).', flush=True)
        return True
    finally:
        handle.close()


def cleanup_incomplete(parent):
    """Recover space from interrupted runs before preprocessing/model loading."""
    parent = Path(parent).resolve()
    if not parent.is_dir():
        return
    for path in parent.iterdir():
        if managed(path) and not complete(path):
            remove_inactive(path, 'incomplete')
