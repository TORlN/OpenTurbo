"""Optional tensor-oriented wrappers above the raw-pointer CUDA API.

The functions in this module accept tensor-like objects with a `data_ptr()`
method, such as PyTorch CUDA tensors. They deliberately use duck typing so the
package does not require PyTorch as an install dependency.
"""

from __future__ import annotations

from typing import Any, Optional

from .cuda_api import (
    encode_tile_fused,
    scan_query_many_cache,
    scan_query_many_cache_multi_tile,
)

_TILE_DIMS = 128
_PACKED_HEADER_BYTES = 32


def _data_ptr(value: Any, name: str) -> int:
    data_ptr = getattr(value, "data_ptr", None)
    if not callable(data_ptr):
        raise TypeError(f"{name} must provide a data_ptr() method")

    ptr = int(data_ptr())
    if ptr == 0:
        raise ValueError(f"{name} returned a null device pointer")
    return ptr


def _is_cuda(value: Any) -> bool:
    attr = getattr(value, "is_cuda", None)
    return True if attr is None else bool(attr)


def _is_contiguous(value: Any) -> bool:
    method = getattr(value, "is_contiguous", None)
    if callable(method):
        return bool(method())
    return True


def _numel(value: Any, name: str) -> int:
    numel_attr = getattr(value, "numel", None)
    if callable(numel_attr):
        return int(numel_attr())
    if isinstance(numel_attr, int):
        return numel_attr
    raise TypeError(f"{name} must provide numel()")


def _nbytes(value: Any, name: str) -> int:
    nbytes_attr = getattr(value, "nbytes", None)
    if isinstance(nbytes_attr, int):
        return nbytes_attr

    numel_attr = getattr(value, "numel", None)
    element_size_attr = getattr(value, "element_size", None)
    if callable(numel_attr) and callable(element_size_attr):
        return int(numel_attr()) * int(element_size_attr())

    raise TypeError(f"{name} must provide nbytes or numel() with element_size()")


def _require_cuda_contiguous(value: Any, name: str) -> None:
    if not _is_cuda(value):
        raise ValueError(f"{name} must be a CUDA-backed tensor-like object")
    if not _is_contiguous(value):
        raise ValueError(f"{name} must be contiguous")


def encode_tile_fused_tensor(
    input_tensor: Any,
    output_headers_tensor: Any,
    token_pos: int,
    rope_theta: float,
    stream_handle: Optional[Any] = None,
) -> None:
    _require_cuda_contiguous(input_tensor, "input_tensor")
    _require_cuda_contiguous(output_headers_tensor, "output_headers_tensor")

    input_numel = _numel(input_tensor, "input_tensor")
    if input_numel % _TILE_DIMS != 0:
        raise ValueError("input_tensor.numel() must be a multiple of 128")

    num_tiles = input_numel // _TILE_DIMS
    required_output_bytes = num_tiles * _PACKED_HEADER_BYTES
    if _nbytes(output_headers_tensor, "output_headers_tensor") < required_output_bytes:
        raise ValueError("output_headers_tensor is too small for the packed headers")

    encode_tile_fused(
        _data_ptr(input_tensor, "input_tensor"),
        _data_ptr(output_headers_tensor, "output_headers_tensor"),
        num_tiles,
        token_pos,
        rope_theta,
        stream_handle,
    )


def scan_query_many_cache_tensor(
    query_header_tensor: Any,
    cache_headers_tensor: Any,
    output_tensor: Any,
    stream_handle: Optional[Any] = None,
) -> None:
    _require_cuda_contiguous(query_header_tensor, "query_header_tensor")
    _require_cuda_contiguous(cache_headers_tensor, "cache_headers_tensor")
    _require_cuda_contiguous(output_tensor, "output_tensor")

    cache_nbytes = _nbytes(cache_headers_tensor, "cache_headers_tensor")
    if cache_nbytes % _PACKED_HEADER_BYTES != 0:
        raise ValueError("cache_headers_tensor byte size must be a multiple of 32")

    num_cache_tiles = cache_nbytes // _PACKED_HEADER_BYTES
    if _nbytes(query_header_tensor, "query_header_tensor") < _PACKED_HEADER_BYTES:
        raise ValueError("query_header_tensor must contain at least one packed header")
    if _nbytes(output_tensor, "output_tensor") < num_cache_tiles * 4:
        raise ValueError("output_tensor is too small for the scan output")

    scan_query_many_cache(
        _data_ptr(query_header_tensor, "query_header_tensor"),
        _data_ptr(cache_headers_tensor, "cache_headers_tensor"),
        _data_ptr(output_tensor, "output_tensor"),
        num_cache_tiles,
        stream_handle,
    )


def scan_query_many_cache_multi_tile_tensor(
    query_headers_tensor: Any,
    cache_headers_tensor: Any,
    output_tensor: Any,
    num_query_tiles: int,
    num_cache_tokens: int,
    stream_handle: Optional[Any] = None,
) -> None:
    _require_cuda_contiguous(query_headers_tensor, "query_headers_tensor")
    _require_cuda_contiguous(cache_headers_tensor, "cache_headers_tensor")
    _require_cuda_contiguous(output_tensor, "output_tensor")

    if num_query_tiles <= 0:
        raise ValueError("num_query_tiles must be positive")
    if num_cache_tokens <= 0:
        raise ValueError("num_cache_tokens must be positive")

    required_query_bytes = num_query_tiles * _PACKED_HEADER_BYTES
    required_cache_bytes = num_query_tiles * num_cache_tokens * _PACKED_HEADER_BYTES
    if _nbytes(query_headers_tensor, "query_headers_tensor") < required_query_bytes:
        raise ValueError("query_headers_tensor is too small for num_query_tiles")
    if _nbytes(cache_headers_tensor, "cache_headers_tensor") < required_cache_bytes:
        raise ValueError("cache_headers_tensor is too small for num_query_tiles * num_cache_tokens")
    if _nbytes(output_tensor, "output_tensor") < num_cache_tokens * 4:
        raise ValueError("output_tensor is too small for the scan output")

    scan_query_many_cache_multi_tile(
        _data_ptr(query_headers_tensor, "query_headers_tensor"),
        _data_ptr(cache_headers_tensor, "cache_headers_tensor"),
        _data_ptr(output_tensor, "output_tensor"),
        num_query_tiles,
        num_cache_tokens,
        stream_handle,
    )
