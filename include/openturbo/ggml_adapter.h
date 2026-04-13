#pragma once

#include "c_api.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define OPENTURBO_GGML_MAX_DIMS 4
#define OPENTURBO_GGML_TYPE_F32 0u
#define OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER 1u

#define OPENTURBO_STREAM_CONTEXT_DEFAULT 0u
#define OPENTURBO_STREAM_CONTEXT_EXTERNAL 1u

    typedef struct openturbo_stream_context_t
    {
        void *cuda_stream;
        uint32_t flags;
    } openturbo_stream_context_t;

    typedef struct openturbo_ggml_tensor_view_t
    {
        void *data;
        uint32_t element_type;
        int32_t n_dims;
        int64_t ne[OPENTURBO_GGML_MAX_DIMS];
        uint64_t nb[OPENTURBO_GGML_MAX_DIMS];
    } openturbo_ggml_tensor_view_t;

    OPENTURBO_CAPI openturbo_status_t openturbo_ggml_encode(
        const openturbo_ggml_tensor_view_t *input,
        const openturbo_ggml_tensor_view_t *output_headers,
        int token_pos,
        float rope_theta,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out);

    OPENTURBO_CAPI openturbo_status_t openturbo_ggml_scan_query_many_cache(
        const openturbo_ggml_tensor_view_t *query_header,
        const openturbo_ggml_tensor_view_t *cache_headers,
        const openturbo_ggml_tensor_view_t *output,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out);

    OPENTURBO_CAPI openturbo_status_t openturbo_ggml_scan_query_many_cache_multi_tile(
        const openturbo_ggml_tensor_view_t *query_headers,
        const openturbo_ggml_tensor_view_t *cache_headers,
        const openturbo_ggml_tensor_view_t *output,
        int num_query_tiles,
        int num_cache_tokens,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out);

#ifdef __cplusplus
}
#endif
