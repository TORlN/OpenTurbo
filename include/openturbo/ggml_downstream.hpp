#pragma once

#include "llama_kv_shim.h"

#ifndef __cplusplus
#error "openturbo/ggml_downstream.hpp requires C++"
#endif

#ifndef GGML_MAX_DIMS
#error "Include ggml.h before openturbo/ggml_downstream.hpp, or provide a compatible ggml_tensor definition"
#endif

struct ggml_tensor;

namespace openturbo::ggml_downstream
{
    inline openturbo_status_t make_view(
        const ggml_tensor *tensor,
        uint32_t element_type,
        openturbo_ggml_tensor_view_t *view_out)
    {
        if (tensor == nullptr || view_out == nullptr)
        {
            return OPENTURBO_STATUS_INVALID_ARGUMENT;
        }

        openturbo_ggml_tensor_view_t view{};
        view.data = tensor->data;
        view.element_type = element_type;
        view.n_dims = OPENTURBO_GGML_MAX_DIMS;

        for (int dim = 0; dim < OPENTURBO_GGML_MAX_DIMS; ++dim)
        {
            view.ne[dim] = tensor->ne[dim];
            view.nb[dim] = static_cast<uint64_t>(tensor->nb[dim]);
        }

        while (view.n_dims > 1 && view.ne[view.n_dims - 1] <= 1)
        {
            --view.n_dims;
        }

        *view_out = view;
        return OPENTURBO_STATUS_SUCCESS;
    }

    inline openturbo_status_t llama_encode_from_ggml_tensors(
        const ggml_tensor *input_heads,
        ggml_tensor *output_headers_by_head,
        int head_index,
        int token_pos,
        float rope_theta,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out)
    {
        if (input_heads == nullptr || output_headers_by_head == nullptr)
        {
            return OPENTURBO_STATUS_INVALID_ARGUMENT;
        }

        if (input_heads->type != GGML_TYPE_F32)
        {
            return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
        }

        openturbo_ggml_tensor_view_t input_view{};
        openturbo_ggml_tensor_view_t output_view{};
        const openturbo_status_t input_status = make_view(input_heads, OPENTURBO_GGML_TYPE_F32, &input_view);
        if (input_status != OPENTURBO_STATUS_SUCCESS)
        {
            return input_status;
        }

        const openturbo_status_t output_status = make_view(output_headers_by_head, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, &output_view);
        if (output_status != OPENTURBO_STATUS_SUCCESS)
        {
            return output_status;
        }

        return openturbo_llama_encode_from_kv_heads(
            &input_view,
            &output_view,
            head_index,
            token_pos,
            rope_theta,
            stream_context,
            cuda_status_out);
    }

    inline openturbo_status_t llama_encode_from_ggml_tensors_prerotated(
        const ggml_tensor *input_heads,
        ggml_tensor *output_headers_by_head,
        int head_index,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out)
    {
        if (input_heads == nullptr || output_headers_by_head == nullptr)
        {
            return OPENTURBO_STATUS_INVALID_ARGUMENT;
        }

        if (input_heads->type != GGML_TYPE_F32)
        {
            return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
        }

        openturbo_ggml_tensor_view_t input_view{};
        openturbo_ggml_tensor_view_t output_view{};
        const openturbo_status_t input_status = make_view(input_heads, OPENTURBO_GGML_TYPE_F32, &input_view);
        if (input_status != OPENTURBO_STATUS_SUCCESS)
        {
            return input_status;
        }

        const openturbo_status_t output_status = make_view(output_headers_by_head, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, &output_view);
        if (output_status != OPENTURBO_STATUS_SUCCESS)
        {
            return output_status;
        }

        return openturbo_llama_encode_from_kv_heads_prerotated(
            &input_view,
            &output_view,
            head_index,
            stream_context,
            cuda_status_out);
    }

    inline openturbo_status_t llama_scan_from_ggml_tensors(
        const ggml_tensor *query_headers_by_head,
        const ggml_tensor *cache_headers_by_head,
        ggml_tensor *output_by_head,
        int head_index,
        int num_query_tiles,
        int num_cache_tokens,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out)
    {
        if (query_headers_by_head == nullptr || cache_headers_by_head == nullptr || output_by_head == nullptr)
        {
            return OPENTURBO_STATUS_INVALID_ARGUMENT;
        }

        if (output_by_head->type != GGML_TYPE_F32)
        {
            return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
        }

        openturbo_ggml_tensor_view_t query_view{};
        openturbo_ggml_tensor_view_t cache_view{};
        openturbo_ggml_tensor_view_t output_view{};
        const openturbo_status_t query_status = make_view(query_headers_by_head, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, &query_view);
        if (query_status != OPENTURBO_STATUS_SUCCESS)
        {
            return query_status;
        }

        const openturbo_status_t cache_status = make_view(cache_headers_by_head, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, &cache_view);
        if (cache_status != OPENTURBO_STATUS_SUCCESS)
        {
            return cache_status;
        }

        const openturbo_status_t output_status = make_view(output_by_head, OPENTURBO_GGML_TYPE_F32, &output_view);
        if (output_status != OPENTURBO_STATUS_SUCCESS)
        {
            return output_status;
        }

        return openturbo_llama_scan_from_kv_cache(
            &query_view,
            &cache_view,
            &output_view,
            head_index,
            num_query_tiles,
            num_cache_tokens,
            stream_context,
            cuda_status_out);
    }

    inline openturbo_status_t llama_encode_all_heads_from_ggml_tensors(
        const ggml_tensor *input_heads,
        ggml_tensor *output_headers_by_head,
        int token_pos,
        float rope_theta,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out)
    {
        if (input_heads == nullptr || output_headers_by_head == nullptr)
        {
            return OPENTURBO_STATUS_INVALID_ARGUMENT;
        }

        if (input_heads->type != GGML_TYPE_F32)
        {
            return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
        }

        openturbo_ggml_tensor_view_t input_view{};
        openturbo_ggml_tensor_view_t output_view{};
        const openturbo_status_t input_status = make_view(input_heads, OPENTURBO_GGML_TYPE_F32, &input_view);
        if (input_status != OPENTURBO_STATUS_SUCCESS)
        {
            return input_status;
        }

        const openturbo_status_t output_status = make_view(output_headers_by_head, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, &output_view);
        if (output_status != OPENTURBO_STATUS_SUCCESS)
        {
            return output_status;
        }

        return openturbo_llama_encode_all_kv_heads(
            &input_view,
            &output_view,
            token_pos,
            rope_theta,
            stream_context,
            cuda_status_out);
    }

    inline openturbo_status_t llama_encode_all_heads_from_ggml_tensors_prerotated(
        const ggml_tensor *input_heads,
        ggml_tensor *output_headers_by_head,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out)
    {
        if (input_heads == nullptr || output_headers_by_head == nullptr)
        {
            return OPENTURBO_STATUS_INVALID_ARGUMENT;
        }

        if (input_heads->type != GGML_TYPE_F32)
        {
            return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
        }

        openturbo_ggml_tensor_view_t input_view{};
        openturbo_ggml_tensor_view_t output_view{};
        const openturbo_status_t input_status = make_view(input_heads, OPENTURBO_GGML_TYPE_F32, &input_view);
        if (input_status != OPENTURBO_STATUS_SUCCESS)
        {
            return input_status;
        }

        const openturbo_status_t output_status = make_view(output_headers_by_head, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, &output_view);
        if (output_status != OPENTURBO_STATUS_SUCCESS)
        {
            return output_status;
        }

        return openturbo_llama_encode_all_kv_heads_prerotated(
            &input_view,
            &output_view,
            stream_context,
            cuda_status_out);
    }

    inline openturbo_status_t llama_scan_all_heads_from_ggml_tensors(
        const ggml_tensor *query_headers_by_head,
        const ggml_tensor *cache_headers_by_head,
        ggml_tensor *output_by_head,
        int num_query_tiles,
        int num_cache_tokens,
        openturbo_stream_context_t stream_context,
        int *cuda_status_out)
    {
        if (query_headers_by_head == nullptr || cache_headers_by_head == nullptr || output_by_head == nullptr)
        {
            return OPENTURBO_STATUS_INVALID_ARGUMENT;
        }

        if (output_by_head->type != GGML_TYPE_F32)
        {
            return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
        }

        openturbo_ggml_tensor_view_t query_view{};
        openturbo_ggml_tensor_view_t cache_view{};
        openturbo_ggml_tensor_view_t output_view{};
        const openturbo_status_t query_status = make_view(query_headers_by_head, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, &query_view);
        if (query_status != OPENTURBO_STATUS_SUCCESS)
        {
            return query_status;
        }

        const openturbo_status_t cache_status = make_view(cache_headers_by_head, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, &cache_view);
        if (cache_status != OPENTURBO_STATUS_SUCCESS)
        {
            return cache_status;
        }

        const openturbo_status_t output_status = make_view(output_by_head, OPENTURBO_GGML_TYPE_F32, &output_view);
        if (output_status != OPENTURBO_STATUS_SUCCESS)
        {
            return output_status;
        }

        return openturbo_llama_scan_all_kv_heads(
            &query_view,
            &cache_view,
            &output_view,
            num_query_tiles,
            num_cache_tokens,
            stream_context,
            cuda_status_out);
    }
}