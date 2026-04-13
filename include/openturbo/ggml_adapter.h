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

#define OPENTURBO_GGML_ENCODE_INPUT_RANK 2
#define OPENTURBO_GGML_ENCODE_OUTPUT_RANK 1
#define OPENTURBO_GGML_SCAN_SINGLE_QUERY_RANK 1
#define OPENTURBO_GGML_SCAN_SINGLE_CACHE_RANK 1
#define OPENTURBO_GGML_SCAN_MULTI_QUERY_RANK 1
#define OPENTURBO_GGML_SCAN_MULTI_CACHE_RANK 2
#define OPENTURBO_GGML_SCAN_OUTPUT_RANK 1

#define OPENTURBO_GGML_DIM_TILE_VALUES 0
#define OPENTURBO_GGML_DIM_TILE_INDEX 1
#define OPENTURBO_GGML_DIM_CACHE_TOKEN 1

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

    /*
     * Encode contract:
     * - input:  rank 2, type F32, shape [128, num_tiles]
     * - output: rank 1, type PACKED_TILE_HEADER, shape [num_tiles]
     *
     * Single-tile scan contract:
     * - query:  rank 1, type PACKED_TILE_HEADER, shape [1]
     * - cache:  rank 1, type PACKED_TILE_HEADER, shape [num_cache_tokens]
     * - output: rank 1, type F32, shape [num_cache_tokens]
     *
     * Multi-tile scan contract:
     * - query:  rank 1, type PACKED_TILE_HEADER, shape [num_query_tiles]
     * - cache:  rank 2, type PACKED_TILE_HEADER, shape [num_query_tiles, num_cache_tokens]
     * - output: rank 1, type F32, shape [num_cache_tokens]
     *
     * All tensors must be contiguous in the declared layout.
     */

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
