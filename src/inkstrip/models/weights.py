"""Download + cache model weights from Hugging Face Hub.

Library users normally don't call this directly — backends call `get_weight()`
on first use. The CLI exposes `inkstrip download-weights` for offline prefetching.

Cache is per-user, OS-appropriate (`platformdirs.user_cache_dir("inkstrip")`).
Override with `InkstripConfig.cache_dir` or `INKSTRIP_CACHE_DIR`.
Set `INKSTRIP_OFFLINE=1` to require already-cached files.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import platformdirs

from inkstrip.models.registry import MODELS, ModelSpec, get_spec
from inkstrip.utils.logging import get_logger

_log = get_logger("weights")


def default_cache_dir() -> Path:
    env = os.environ.get("INKSTRIP_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(platformdirs.user_cache_dir("inkstrip"))


def _is_offline(offline: bool) -> bool:
    return offline or os.environ.get("INKSTRIP_OFFLINE", "").lower() in {"1", "true", "yes"}


def get_weight(
    name: str,
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> Path:
    """Resolve a registered model name to a local file path, downloading if needed."""
    spec = get_spec(name)
    return _fetch(spec, cache_dir=cache_dir, offline=offline)


def prefetch_all(
    *,
    cache_dir: Path | None = None,
    names: list[str] | None = None,
) -> dict[str, Path]:
    """Download all (or named) model weights up-front. Returns name → path."""
    targets = names or list(MODELS.keys())
    out: dict[str, Path] = {}
    for n in targets:
        out[n] = _fetch(get_spec(n), cache_dir=cache_dir, offline=False)
    return out


def _fetch(spec: ModelSpec, *, cache_dir: Path | None, offline: bool) -> Path:
    cache = (cache_dir or default_cache_dir()).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    kwargs = dict(
        repo_id=spec.repo,
        filename=spec.filename,
        revision=spec.revision,
        cache_dir=str(cache),
    )
    if _is_offline(offline):
        kwargs["local_files_only"] = True

    _log.debug("resolving %s from %s@%s", spec.filename, spec.repo, spec.revision)
    path = Path(hf_hub_download(**kwargs))

    if spec.sha256:
        actual = _sha256(path)
        if actual != spec.sha256:
            raise RuntimeError(
                f"sha256 mismatch for {spec.name}: expected {spec.sha256}, got {actual}"
            )
    return path


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()
