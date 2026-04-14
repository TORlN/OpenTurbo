#ifndef OPENTURBO_CAPI_EXPORTS
#define OPENTURBO_CAPI_EXPORTS
#endif

#include "../include/openturbo/c_api.h"

#include "openturbo_cuda_api.cuh"

#include <cuda_runtime.h>

static_assert(sizeof(openturbo_packed_tile_header_t) == sizeof(openturbo::PackedTileHeader));

namespace
{
    inline openturbo_status_t export_cuda_status(cudaError_t cuda_status, int *cuda_status_out)
    {
        if (cuda_status_out != nullptr)
        {
            *cuda_status_out = static_cast<int>(cuda_status);
        }

        return (cuda_status == cudaSuccess) ? OPENTURBO_STATUS_SUCCESS : OPENTURBO_STATUS_CUDA_ERROR;
    }

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

extern "C" OPENTURBO_CAPI uint32_t openturbo_get_c_api_version(void)
{
    return OPENTURBO_CAPI_VERSION;
}

extern "C" OPENTURBO_CAPI const char *openturbo_status_string(int status)
{
    switch (static_cast<openturbo_status_t>(status))
    {
    case OPENTURBO_STATUS_SUCCESS:
        return "success";
    case OPENTURBO_STATUS_INVALID_ARGUMENT:
        return "invalid_argument";
    case OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT:
        return "incompatible_layout";
    case OPENTURBO_STATUS_CUDA_ERROR:
        return "cuda_error";
    default:
        return "unknown_status";
    }
}

extern "C" OPENTURBO_CAPI openturbo_status_t openturbo_encode_tile_fused(
    const float *input,
    openturbo_packed_tile_header_t *output_headers,
    int num_tiles,
    int token_pos,
    float rope_theta,
    void *stream_handle,
    int *cuda_status_out)
{
    if (input == nullptr || output_headers == nullptr || num_tiles <= 0)
    {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    return export_cuda_status(openturbo::launch_encode_tile_fused(
                                  input,
                                  as_cpp_headers(output_headers),
                                  num_tiles,
                                  token_pos,
                                  rope_theta,
                                  normalize_stream_handle(stream_handle)),
                              cuda_status_out);
}

extern "C" OPENTURBO_CAPI openturbo_status_t openturbo_encode_tile_fused_prerotated(
    const float *input,
    openturbo_packed_tile_header_t *output_headers,
    int num_tiles,
    void *stream_handle,
    int *cuda_status_out)
{
    if (input == nullptr || output_headers == nullptr || num_tiles <= 0)
    {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    return export_cuda_status(openturbo::launch_encode_tile_fused_prerotated(
                                  input,
                                  as_cpp_headers(output_headers),
                                  num_tiles,
                                  normalize_stream_handle(stream_handle)),
                              cuda_status_out);
}

extern "C" OPENTURBO_CAPI openturbo_status_t openturbo_encode_tile_fused_f16(
    const uint16_t *input,
    openturbo_packed_tile_header_t *output_headers,
    int num_tiles,
    int token_pos,
    float rope_theta,
    void *stream_handle,
    int *cuda_status_out)
{
    if (input == nullptr || output_headers == nullptr || num_tiles <= 0)
    {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    return export_cuda_status(openturbo::launch_encode_tile_fused(
                                  reinterpret_cast<const __half *>(input),
                                  as_cpp_headers(output_headers),
                                  num_tiles,
                                  token_pos,
                                  rope_theta,
                                  normalize_stream_handle(stream_handle)),
                              cuda_status_out);
}

extern "C" OPENTURBO_CAPI openturbo_status_t openturbo_encode_tile_fused_prerotated_f16(
    const uint16_t *input,
    openturbo_packed_tile_header_t *output_headers,
    int num_tiles,
    void *stream_handle,
    int *cuda_status_out)
{
    if (input == nullptr || output_headers == nullptr || num_tiles <= 0)
    {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    return export_cuda_status(openturbo::launch_encode_tile_fused_prerotated(
                                  reinterpret_cast<const __half *>(input),
                                  as_cpp_headers(output_headers),
                                  num_tiles,
                                  normalize_stream_handle(stream_handle)),
                              cuda_status_out);
}

extern "C" OPENTURBO_CAPI openturbo_status_t openturbo_scan_query_many_cache(
    const openturbo_packed_tile_header_t *query_header,
    const openturbo_packed_tile_header_t *cache_headers,
    float *output,
    int num_cache_tiles,
    void *stream_handle,
    int *cuda_status_out)
{
    if (query_header == nullptr || cache_headers == nullptr || output == nullptr || num_cache_tiles <= 0)
    {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    return export_cuda_status(openturbo::launch_scan_query_many_cache(
                                  as_cpp_headers(query_header),
                                  as_cpp_headers(cache_headers),
                                  output,
                                  num_cache_tiles,
                                  normalize_stream_handle(stream_handle)),
                              cuda_status_out);
}

extern "C" OPENTURBO_CAPI openturbo_status_t openturbo_scan_query_many_cache_multi_tile(
    const openturbo_packed_tile_header_t *query_headers,
    const openturbo_packed_tile_header_t *cache_headers,
    float *output,
    int num_query_tiles,
    int num_cache_tokens,
    void *stream_handle,
    int *cuda_status_out)
{
    if (query_headers == nullptr || cache_headers == nullptr || output == nullptr || num_query_tiles <= 0 || num_cache_tokens <= 0)
    {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    return export_cuda_status(openturbo::launch_scan_query_many_cache_multi_tile(
                                  as_cpp_headers(query_headers),
                                  as_cpp_headers(cache_headers),
                                  output,
                                  num_query_tiles,
                                  num_cache_tokens,
                                  normalize_stream_handle(stream_handle)),
                              cuda_status_out);
}

extern "C" OPENTURBO_CAPI const char *openturbo_cuda_error_string(int status)
{
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}
