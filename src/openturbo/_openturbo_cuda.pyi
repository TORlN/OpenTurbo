from typing import Any

def encode_tile_fused(
    input_ptr: int,
    output_headers_ptr: int,
    num_tiles: int,
    token_pos: int,
    rope_theta: float,
    stream_handle: int = 0,
) -> None: ...

def scan_query_many_cache(
    query_header_ptr: int,
    cache_headers_ptr: int,
    output_ptr: int,
    num_cache_tiles: int,
    stream_handle: int = 0,
) -> None: ...

def scan_query_many_cache_multi_tile(
    query_headers_ptr: int,
    cache_headers_ptr: int,
    output_ptr: int,
    num_query_tiles: int,
    num_cache_tokens: int,
    stream_handle: int = 0,
) -> None: ...