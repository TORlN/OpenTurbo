#ifndef OPENTURBO_CAPI_EXPORTS
#define OPENTURBO_CAPI_EXPORTS
#endif

#include "../include/openturbo/llama_bridge.h"

namespace
{
    bool has_supported_layout(uint32_t layout)
    {
        return layout == OPENTURBO_LLAMA_LAYOUT_FLAT_TILES;
    }
}

extern "C" OPENTURBO_CAPI openturbo_status_t openturbo_llama_encode(
    const openturbo_llama_encode_request_t *request,
    int *cuda_status_out)
{
    if (request == nullptr)
    {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    if (!has_supported_layout(request->layout))
    {
        return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
    }

    return openturbo_ggml_encode(
        &request->input,
        &request->output_headers,
        request->token_pos,
        request->rope_theta,
        request->stream_context,
        cuda_status_out);
}

extern "C" OPENTURBO_CAPI openturbo_status_t openturbo_llama_scan(
    const openturbo_llama_scan_request_t *request,
    int *cuda_status_out)
{
    if (request == nullptr)
    {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    if (!has_supported_layout(request->layout))
    {
        return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
    }

    if (request->num_query_tiles == 1)
    {
        return openturbo_ggml_scan_query_many_cache(
            &request->query_headers,
            &request->cache_headers,
            &request->output,
            request->stream_context,
            cuda_status_out);
    }

    return openturbo_ggml_scan_query_many_cache_multi_tile(
        &request->query_headers,
        &request->cache_headers,
        &request->output,
        request->num_query_tiles,
        request->num_cache_tokens,
        request->stream_context,
        cuda_status_out);
}
