"""Python-facing wrappers for the CUDA extension.

This module defines the stable Python contract around the raw-pointer launch
wrappers exposed by the compiled `_openturbo_cuda` extension. The extension may
still be unavailable during early development; these helpers fail with a clear
error when the binary module has not been built yet.
"""

import os
from pathlib import Path
from typing import Any, Optional


def _configure_windows_dll_search() -> None:
    if os.name != "nt":
        return

    candidates = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")

    candidates.append(Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"))

    for directory in candidates:
        if not directory.is_dir():
            continue

        try:
            os.add_dll_directory(str(directory))
        except (AttributeError, FileNotFoundError, OSError):
            continue


_configure_windows_dll_search()

try:
    from . import _openturbo_cuda as _native
except ImportError:  # pragma: no cover - exercised only before the extension exists
    _native = None


def is_cuda_extension_available() -> bool:
    """Return whether the compiled CUDA extension is importable."""
    return _native is not None


def require_cuda_extension() -> None:
    """Raise a clear error if the compiled CUDA extension is unavailable."""
    if _native is None:
        raise RuntimeError(
            "The OpenTurbo CUDA extension is not built yet. "
            "Build the compiled module before calling CUDA launch wrappers."
        )


def _coerce_ptr(value: Any, name: str) -> int:
    """Normalize a raw pointer-like value to Python int.

    The intended callers are objects that already expose device pointers as
    integers, such as `torch.Tensor.data_ptr()` or raw integer addresses.
    """
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer-like device pointer") from exc


def encode_tile_fused(
    input_ptr: Any,
    output_headers_ptr: Any,
    num_tiles: int,
    token_pos: int,
    rope_theta: float,
    stream_handle: Optional[Any] = None,
) -> None:
    """Launch the fused encoder on raw device pointers."""
    require_cuda_extension()
    _native.encode_tile_fused(
        _coerce_ptr(input_ptr, "input_ptr"),
        _coerce_ptr(output_headers_ptr, "output_headers_ptr"),
        int(num_tiles),
        int(token_pos),
        float(rope_theta),
        0 if stream_handle is None else _coerce_ptr(stream_handle, "stream_handle"),
    )


def scan_query_many_cache(
    query_header_ptr: Any,
    cache_headers_ptr: Any,
    output_ptr: Any,
    num_cache_tiles: int,
    stream_handle: Optional[Any] = None,
) -> None:
    """Launch the one-query-header versus many-cache-headers scan kernel."""
    require_cuda_extension()
    _native.scan_query_many_cache(
        _coerce_ptr(query_header_ptr, "query_header_ptr"),
        _coerce_ptr(cache_headers_ptr, "cache_headers_ptr"),
        _coerce_ptr(output_ptr, "output_ptr"),
        int(num_cache_tiles),
        0 if stream_handle is None else _coerce_ptr(stream_handle, "stream_handle"),
    )


def scan_query_many_cache_multi_tile(
    query_headers_ptr: Any,
    cache_headers_ptr: Any,
    output_ptr: Any,
    num_query_tiles: int,
    num_cache_tokens: int,
    stream_handle: Optional[Any] = None,
) -> None:
    """Launch the multi-tile query head versus many cache tokens scan kernel."""
    require_cuda_extension()
    _native.scan_query_many_cache_multi_tile(
        _coerce_ptr(query_headers_ptr, "query_headers_ptr"),
        _coerce_ptr(cache_headers_ptr, "cache_headers_ptr"),
        _coerce_ptr(output_ptr, "output_ptr"),
        int(num_query_tiles),
        int(num_cache_tokens),
        0 if stream_handle is None else _coerce_ptr(stream_handle, "stream_handle"),
    )