#pragma once

#include "ggml_adapter.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define OPENTURBO_LLAMA_LAYOUT_FLAT_TILES 0u

    typedef struct openturbo_llama_encode_request_t
    {
        openturbo_ggml_tensor_view_t input;
        openturbo_ggml_tensor_view_t output_headers;
        int token_pos;
        float rope_theta;
        openturbo_stream_context_t stream_context;
        uint32_t layout;
    } openturbo_llama_encode_request_t;

    typedef struct openturbo_llama_scan_request_t
    {
        openturbo_ggml_tensor_view_t query_headers;
        openturbo_ggml_tensor_view_t cache_headers;
        openturbo_ggml_tensor_view_t output;
        int num_query_tiles;
        int num_cache_tokens;
        openturbo_stream_context_t stream_context;
        uint32_t layout;
    } openturbo_llama_scan_request_t;

    OPENTURBO_CAPI openturbo_status_t openturbo_llama_encode(
        const openturbo_llama_encode_request_t *request,
        int *cuda_status_out);

    OPENTURBO_CAPI openturbo_status_t openturbo_llama_scan(
        const openturbo_llama_scan_request_t *request,
        int *cuda_status_out);

#ifdef __cplusplus
}
#endif
