"""Thin tqdm wrapper with a no-op fallback.

The upstream Triangle Splatting+/3DGS training uses ``tqdm`` for its progress
bar; we mirror that here. If ``tqdm`` is not installed the ``progress`` helper
returns a no-op object exposing the same (``set_postfix``/``update``/``close``)
interface, so the training/convert/render loops run unchanged. Import-safe with
no torch dependency, so it is available in every environment.
"""

from __future__ import annotations

try:  # tqdm ships with the baseline deps on the GPU server
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - only hit when tqdm is absent
    _tqdm = None


class _NoBar:
    """Fallback iterable matching the subset of the tqdm API we use."""

    def __init__(self, iterable=None, **_):
        self._it = [] if iterable is None else iterable

    def __iter__(self):
        return iter(self._it)

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass


def progress(iterable=None, **kwargs):
    """Return a tqdm bar over ``iterable`` (or a no-op bar if tqdm is missing)."""
    if _tqdm is None:
        return _NoBar(iterable)
    return _tqdm(iterable, **kwargs)
