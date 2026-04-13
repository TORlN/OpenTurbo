#include "../include/openturbo/c_api.h"

#include "openturbo_cuda_api.cuh"

#include <cuda_runtime.h>

static_assert(sizeof(openturbo_packed_tile_header_t) == sizeof(openturbo::PackedTileHeader));

namespace
{
    inline cudaStream_t normalize_stream_handle(void *stream_handle)
    {
        return reinterpret_cast<cudaStream_t>(stream_handle);
    }

    inline openturbo::PackedTileHeader *as_cpp_headers(openturbo_packed_tile_header_t *headers)
    {
        return reinterpret_cast<openturbo::PackedTileHeader *>(headers);
    }

    inline const openturbo::PackedTileHeader *as_cpp_headers(const openturbo_packed_tile_header_t *headers)
    {
        return reinterpret_cast<const openturbo::PackedTileHeader *>(headers);
    }
}

extern "C" OPENTURBO_CAPI int openturbo_encode_tile_fused(
    const float *input,
    openturbo_packed_tile_header_t *output_headers,
    int num_tiles,
    int token_pos,
    float rope_theta,
    void *stream_handle)
{
    return static_cast<int>(openturbo::launch_encode_tile_fused(
        input,
        as_cpp_headers(output_headers),
        num_tiles,
        token_pos,
        rope_theta,
        normalize_stream_handle(stream_handle)));
}

extern "C" OPENTURBO_CAPI int openturbo_scan_query_many_cache(
    const openturbo_packed_tile_header_t *query_header,
    const openturbo_packed_tile_header_t *cache_headers,
    float *output,
    int num_cache_tiles,
    void *stream_handle)
{
    return static_cast<int>(openturbo::launch_scan_query_many_cache(
        as_cpp_headers(query_header),
        as_cpp_headers(cache_headers),
        output,
        num_cache_tiles,
        normalize_stream_handle(stream_handle)));
}

extern "C" OPENTURBO_CAPI int openturbo_scan_query_many_cache_multi_tile(
    const openturbo_packed_tile_header_t *query_headers,
    const openturbo_packed_tile_header_t *cache_headers,
    float *output,
    int num_query_tiles,
    int num_cache_tokens,
    void *stream_handle)
{
    return static_cast<int>(openturbo::launch_scan_query_many_cache_multi_tile(
        as_cpp_headers(query_headers),
        as_cpp_headers(cache_headers),
        output,
        num_query_tiles,
        num_cache_tokens,
        normalize_stream_handle(stream_handle)));
}

extern "C" OPENTURBO_CAPI const char *openturbo_cuda_error_string(int status)
{
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}
