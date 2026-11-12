"""Automatic file-change detection (no extra dependency).

Polls the modification time + size of each file every `interval` seconds. When
something changes it waits `debounce` seconds (editors often save in two
steps), then compares a SHA-256 of the contents so a save that did not change
the data does not trigger a re-run.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable, Iterable


def content_hash(paths: Iterable[Path]) -> str | None:
    """SHA-256 over all files; None if any file is currently missing/unreadable."""
    h = hashlib.sha256()
    for p in paths:
        try:
            h.update(Path(p).name.encode())
            h.update(Path(p).read_bytes())
        except OSError:
            return None
    return h.hexdigest()


def _signature(paths: Iterable[Path]):
    sig = []
    for p in paths:
        try:
            st = Path(p).stat()
            sig.append((st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append(None)
    return tuple(sig)


def watch(paths: list[Path], on_change: Callable[[], None], interval: float = 2.0,
          debounce: float = 1.0, stop: Callable[[], bool] | None = None) -> None:
    last_sig = _signature(paths)
    last_hash = content_hash(paths)
    while not (stop and stop()):
        time.sleep(interval)
        sig = _signature(paths)
        if sig == last_sig:
            continue
        time.sleep(debounce)              # let the editor finish writing
        sig = _signature(paths)
        new_hash = content_hash(paths)
        if new_hash is None:              # file briefly missing mid-save: try again next tick
            continue
        last_sig = sig
        if new_hash == last_hash:         # touched but identical content
            continue
        last_hash = new_hash
        try:
            on_change()
        except Exception as exc:          # a half-edited CSV must not kill the watcher
            print(f"[watch] Run failed: {exc}. Fix the file and save again.")
