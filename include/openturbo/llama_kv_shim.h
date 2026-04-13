#pragma once

#include "llama_bridge.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define OPENTURBO_LLAMA_SHIM_ENCODE_INPUT_RANK 3
#define OPENTURBO_LLAMA_SHIM_ENCODE_OUTPUT_RANK 2
#define OPENTURBO_LLAMA_SHIM_SCAN_QUERY_RANK 2
#define OPENTURBO_LLAMA_SHIM_SCAN_CACHE_RANK 3
#define OPENTURBO_LLAMA_SHIM_SCAN_OUTPUT_RANK 2

#define OPENTURBO_LLAMA_SHIM_DIM_VALUES 0
#define OPENTURBO_LLAMA_SHIM_DIM_HEAD_TILE 1
#define OPENTURBO_LLAMA_SHIM_DIM_CACHE_TOKEN 1
#define OPENTURBO_LLAMA_SHIM_DIM_HEAD_INDEX 2

    /*
     * Llama KV shim contract:
     *
     * Encode source storage:
     * - input:  rank 3, type F32, shape [128, num_head_tiles, num_heads]
     * - output: rank 2, type PACKED_TILE_HEADER, shape [num_head_tiles, num_heads]
     *
     * Scan source storage:
     * - query:  rank 2, type PACKED_TILE_HEADER, shape [num_query_tiles, num_heads]
     * - cache:  rank 3, type PACKED_TILE_HEADER, shape [num_query_tiles, num_cache_tokens, num_heads]
     * - output: rank 2, type F32, shape [num_cache_tokens, num_heads]
     *
     * The shim can either slice one head index from these dense tensors or iterate all
     * heads and forward the resulting head-local views into the lower-level llama bridge.
     */

    OPENTURBO_CAPI openturbo_status_t openturbo_llama_encode_from_kv_heads(
        const openturbo_ggml_tensor_view_t *input_heads,
        const openturbo_ggml_tensor_view_t *output_headers_by_head,
        int head_index,
        int token_pos,
        float rope_theta,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out);

    OPENTURBO_CAPI openturbo_status_t openturbo_llama_encode_from_kv_heads_prerotated(
        const openturbo_ggml_tensor_view_t *input_heads,
        const openturbo_ggml_tensor_view_t *output_headers_by_head,
        int head_index,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out);

    OPENTURBO_CAPI openturbo_status_t openturbo_llama_scan_from_kv_cache(
        const openturbo_ggml_tensor_view_t *query_headers_by_head,
        const openturbo_ggml_tensor_view_t *cache_headers_by_head,
        const openturbo_ggml_tensor_view_t *output_by_head,
        int head_index,
        int num_query_tiles,
        int num_cache_tokens,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out);

    OPENTURBO_CAPI openturbo_status_t openturbo_llama_encode_all_kv_heads(
        const openturbo_ggml_tensor_view_t *input_heads,
        const openturbo_ggml_tensor_view_t *output_headers_by_head,
        int token_pos,
        float rope_theta,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out);

    OPENTURBO_CAPI openturbo_status_t openturbo_llama_encode_all_kv_heads_prerotated(
        const openturbo_ggml_tensor_view_t *input_heads,
        const openturbo_ggml_tensor_view_t *output_headers_by_head,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out);

    OPENTURBO_CAPI openturbo_status_t openturbo_llama_scan_all_kv_heads(
        const openturbo_ggml_tensor_view_t *query_headers_by_head,
        const openturbo_ggml_tensor_view_t *cache_headers_by_head,
        const openturbo_ggml_tensor_view_t *output_by_head,
        int num_query_tiles,
        int num_cache_tokens,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out);

#ifdef __cplusplus
}
#endif