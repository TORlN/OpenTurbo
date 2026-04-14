#pragma once

#include <stdint.h>

#ifdef _WIN32
#if defined(OPENTURBO_CAPI_EXPORTS)
#define OPENTURBO_CAPI __declspec(dllexport)
#else
#define OPENTURBO_CAPI __declspec(dllimport)
#endif
#else
#define OPENTURBO_CAPI
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#define OPENTURBO_CAPI_VERSION_MAJOR 1u
#define OPENTURBO_CAPI_VERSION_MINOR 0u
#define OPENTURBO_CAPI_VERSION_PATCH 0u
#define OPENTURBO_CAPI_VERSION ((OPENTURBO_CAPI_VERSION_MAJOR << 24) | (OPENTURBO_CAPI_VERSION_MINOR << 16) | OPENTURBO_CAPI_VERSION_PATCH)
#define OPENTURBO_TILE_DIMS 128
#define OPENTURBO_PACKED_TILE_HEADER_BYTES 32

    typedef enum openturbo_status_t
    {
        OPENTURBO_STATUS_SUCCESS = 0,
        OPENTURBO_STATUS_INVALID_ARGUMENT = 1,
        OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT = 2,
        OPENTURBO_STATUS_CUDA_ERROR = 3
    } openturbo_status_t;

    typedef struct openturbo_packed_tile_header_t
    {
        uint64_t quadrant_word_0;
        uint64_t quadrant_word_1;
        uint64_t qjl_sign_word;
        uint16_t block_scale_fp16_bits;
        uint16_t local_alpha_fp16_bits;
        uint32_t reserved_u32;
    } openturbo_packed_tile_header_t;

    OPENTURBO_CAPI uint32_t openturbo_get_c_api_version(void);

    OPENTURBO_CAPI const char *openturbo_status_string(int status);

    OPENTURBO_CAPI openturbo_status_t openturbo_encode_tile_fused(
        const float *input,
        openturbo_packed_tile_header_t *output_headers,
        int num_tiles,
        int token_pos,
        float rope_theta,
        void *stream_handle,
        int *cuda_status_out);

    OPENTURBO_CAPI openturbo_status_t openturbo_encode_tile_fused_prerotated(
        const float *input,
        openturbo_packed_tile_header_t *output_headers,
        int num_tiles,
        void *stream_handle,
        int *cuda_status_out);

    OPENTURBO_CAPI openturbo_status_t openturbo_encode_tile_fused_f16(
        const uint16_t *input,
        openturbo_packed_tile_header_t *output_headers,
        int num_tiles,
        int token_pos,
        float rope_theta,
        void *stream_handle,
        int *cuda_status_out);

    OPENTURBO_CAPI openturbo_status_t openturbo_encode_tile_fused_prerotated_f16(
        const uint16_t *input,
        openturbo_packed_tile_header_t *output_headers,
        int num_tiles,
        void *stream_handle,
        int *cuda_status_out);

    OPENTURBO_CAPI openturbo_status_t openturbo_scan_query_many_cache(
        const openturbo_packed_tile_header_t *query_header,
        const openturbo_packed_tile_header_t *cache_headers,
        float *output,
        int num_cache_tiles,
        void *stream_handle,
        int *cuda_status_out);

    OPENTURBO_CAPI openturbo_status_t openturbo_scan_query_many_cache_multi_tile(
        const openturbo_packed_tile_header_t *query_headers,
        const openturbo_packed_tile_header_t *cache_headers,
        float *output,
        int num_query_tiles,
        int num_cache_tokens,
        void *stream_handle,
        int *cuda_status_out);

    OPENTURBO_CAPI const char *openturbo_cuda_error_string(int cuda_status);

#ifdef __cplusplus
}

static_assert(sizeof(openturbo_packed_tile_header_t) == OPENTURBO_PACKED_TILE_HEADER_BYTES);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(openturbo_packed_tile_header_t) == OPENTURBO_PACKED_TILE_HEADER_BYTES, "packed header size must stay 32 bytes");
#endif
