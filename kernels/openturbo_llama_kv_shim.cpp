#ifndef OPENTURBO_CAPI_EXPORTS
#define OPENTURBO_CAPI_EXPORTS
#endif

#include "../include/openturbo/llama_kv_shim.h"

#include <stddef.h>

namespace
{
    uint64_t element_stride_bytes(uint32_t element_type)
    {
        switch (element_type)
        {
        case OPENTURBO_GGML_TYPE_F32:
            return 4;
        case OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER:
            return OPENTURBO_PACKED_TILE_HEADER_BYTES;
        default:
            return 0;
        }
    }

    bool has_rank(const openturbo_ggml_tensor_view_t &view, int32_t expected_rank)
    {
        return view.n_dims == expected_rank;
    }

    bool is_contiguous_dense(const openturbo_ggml_tensor_view_t &view)
    {
        if (view.data == nullptr || view.n_dims <= 0 || view.n_dims > OPENTURBO_GGML_MAX_DIMS)
        {
            return false;
        }

        uint64_t expected_stride = element_stride_bytes(view.element_type);
        if (expected_stride == 0)
        {
            return false;
        }

        for (int dim = 0; dim < view.n_dims; ++dim)
        {
            if (view.ne[dim] <= 0 || view.nb[dim] != expected_stride)
            {
                return false;
            }
            expected_stride *= static_cast<uint64_t>(view.ne[dim]);
        }

        return true;
    }

    bool is_encode_input_storage(const openturbo_ggml_tensor_view_t &view)
    {
        return view.element_type == OPENTURBO_GGML_TYPE_F32 &&
               has_rank(view, OPENTURBO_LLAMA_SHIM_ENCODE_INPUT_RANK) &&
               view.ne[OPENTURBO_LLAMA_SHIM_DIM_VALUES] == OPENTURBO_TILE_DIMS &&
               view.ne[OPENTURBO_LLAMA_SHIM_DIM_HEAD_TILE] > 0 &&
               view.ne[OPENTURBO_LLAMA_SHIM_DIM_HEAD_INDEX] > 0 &&
               is_contiguous_dense(view);
    }

    bool is_encode_output_storage(const openturbo_ggml_tensor_view_t &view)
    {
        return view.element_type == OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER &&
               has_rank(view, OPENTURBO_LLAMA_SHIM_ENCODE_OUTPUT_RANK) &&
               view.ne[0] > 0 &&
               view.ne[1] > 0 &&
               is_contiguous_dense(view);
    }

    bool is_scan_query_storage(const openturbo_ggml_tensor_view_t &view)
    {
        return view.element_type == OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER &&
               has_rank(view, OPENTURBO_LLAMA_SHIM_SCAN_QUERY_RANK) &&
               view.ne[0] > 0 &&
               view.ne[1] > 0 &&
               is_contiguous_dense(view);
    }

    bool is_scan_cache_storage(const openturbo_ggml_tensor_view_t &view)
    {
        return view.element_type == OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER &&
               has_rank(view, OPENTURBO_LLAMA_SHIM_SCAN_CACHE_RANK) &&
               view.ne[0] > 0 &&
               view.ne[1] > 0 &&
               view.ne[2] > 0 &&
               is_contiguous_dense(view);
    }

    bool is_scan_output_storage(const openturbo_ggml_tensor_view_t &view)
    {
        return view.element_type == OPENTURBO_GGML_TYPE_F32 &&
               has_rank(view, OPENTURBO_LLAMA_SHIM_SCAN_OUTPUT_RANK) &&
               view.ne[0] > 0 &&
               view.ne[1] > 0 &&
               is_contiguous_dense(view);
    }

    bool has_valid_head_index(int head_index, int64_t num_heads)
    {
        return head_index >= 0 && static_cast<int64_t>(head_index) < num_heads;
    }

    void *offset_data(void *data, uint64_t offset_bytes)
    {
        return static_cast<void *>(static_cast<unsigned char *>(data) + offset_bytes);
    }

    const void *offset_data(const void *data, uint64_t offset_bytes)
    {
        return static_cast<const void *>(static_cast<const unsigned char *>(data) + offset_bytes);
    }

    openturbo_ggml_tensor_view_t make_encode_input_head_view(const openturbo_ggml_tensor_view_t &input_heads, int head_index)
    {
        openturbo_ggml_tensor_view_t head_view{};
        head_view.data = const_cast<void *>(offset_data(input_heads.data, static_cast<uint64_t>(head_index) * input_heads.nb[2]));
        head_view.element_type = input_heads.element_type;
        head_view.n_dims = OPENTURBO_GGML_ENCODE_INPUT_RANK;
        head_view.ne[0] = input_heads.ne[0];
        head_view.ne[1] = input_heads.ne[1];
        head_view.nb[0] = input_heads.nb[0];
        head_view.nb[1] = input_heads.nb[1];
        return head_view;
    }

    openturbo_ggml_tensor_view_t make_encode_output_head_view(const openturbo_ggml_tensor_view_t &output_headers_by_head, int head_index)
    {
        openturbo_ggml_tensor_view_t head_view{};
        head_view.data = offset_data(output_headers_by_head.data, static_cast<uint64_t>(head_index) * output_headers_by_head.nb[1]);
        head_view.element_type = output_headers_by_head.element_type;
        head_view.n_dims = OPENTURBO_GGML_ENCODE_OUTPUT_RANK;
        head_view.ne[0] = output_headers_by_head.ne[0];
        head_view.nb[0] = output_headers_by_head.nb[0];
        return head_view;
    }

    openturbo_ggml_tensor_view_t make_scan_query_head_view(const openturbo_ggml_tensor_view_t &query_headers_by_head, int head_index)
    {
        openturbo_ggml_tensor_view_t head_view{};
        head_view.data = const_cast<void *>(offset_data(query_headers_by_head.data, static_cast<uint64_t>(head_index) * query_headers_by_head.nb[1]));
        head_view.element_type = query_headers_by_head.element_type;
        head_view.n_dims = OPENTURBO_GGML_SCAN_MULTI_QUERY_RANK;
        head_view.ne[0] = query_headers_by_head.ne[0];
        head_view.nb[0] = query_headers_by_head.nb[0];
        return head_view;
    }

    openturbo_ggml_tensor_view_t make_scan_cache_head_view(const openturbo_ggml_tensor_view_t &cache_headers_by_head, int head_index)
    {
        openturbo_ggml_tensor_view_t head_view{};
        head_view.data = const_cast<void *>(offset_data(cache_headers_by_head.data, static_cast<uint64_t>(head_index) * cache_headers_by_head.nb[2]));
        head_view.element_type = cache_headers_by_head.element_type;
        head_view.n_dims = OPENTURBO_GGML_SCAN_MULTI_CACHE_RANK;
        head_view.ne[0] = cache_headers_by_head.ne[0];
        head_view.ne[1] = cache_headers_by_head.ne[1];
        head_view.nb[0] = cache_headers_by_head.nb[0];
        head_view.nb[1] = cache_headers_by_head.nb[1];
        return head_view;
    }

    openturbo_ggml_tensor_view_t make_scan_output_head_view(const openturbo_ggml_tensor_view_t &output_by_head, int head_index)
    {
        openturbo_ggml_tensor_view_t head_view{};
        head_view.data = const_cast<void *>(offset_data(output_by_head.data, static_cast<uint64_t>(head_index) * output_by_head.nb[1]));
        head_view.element_type = output_by_head.element_type;
        head_view.n_dims = OPENTURBO_GGML_SCAN_OUTPUT_RANK;
        head_view.ne[0] = output_by_head.ne[0];
        head_view.nb[0] = output_by_head.nb[0];
        return head_view;
    }
}

extern "C" OPENTURBO_CAPI openturbo_status_t openturbo_llama_encode_from_kv_heads(
    const openturbo_ggml_tensor_view_t *input_heads,
    const openturbo_ggml_tensor_view_t *output_headers_by_head,
    int head_index,
    int token_pos,
    float rope_theta,
    openturbo_stream_context_t stream_context,
    int *cuda_status_out)
{
    if (input_heads == nullptr || output_headers_by_head == nullptr || !has_valid_head_index(head_index, input_heads->ne[2]))
    {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    if (!is_encode_input_storage(*input_heads) ||
        !is_encode_output_storage(*output_headers_by_head) ||
        output_headers_by_head->ne[0] != input_heads->ne[1] ||
        output_headers_by_head->ne[1] != input_heads->ne[2])
    {
        return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
    }

    const openturbo_ggml_tensor_view_t input_head_view = make_encode_input_head_view(*input_heads, head_index);
    const openturbo_ggml_tensor_view_t output_head_view = make_encode_output_head_view(*output_headers_by_head, head_index);
    const openturbo_llama_encode_request_t request{
        input_head_view,
        output_head_view,
        token_pos,
        rope_theta,
        stream_context,
        OPENTURBO_LLAMA_LAYOUT_HEAD_LOCAL_KV_TILES_V1};
    return openturbo_llama_encode(&request, cuda_status_out);
}

extern "C" OPENTURBO_CAPI openturbo_status_t openturbo_llama_scan_from_kv_cache(
    const openturbo_ggml_tensor_view_t *query_headers_by_head,
    const openturbo_ggml_tensor_view_t *cache_headers_by_head,
    const openturbo_ggml_tensor_view_t *output_by_head,
    int head_index,
    int num_query_tiles,
    int num_cache_tokens,
    openturbo_stream_context_t stream_context,
    int *cuda_status_out)
{
    if (query_headers_by_head == nullptr || cache_headers_by_head == nullptr || output_by_head == nullptr ||
        num_query_tiles <= 0 || num_cache_tokens <= 0 || !has_valid_head_index(head_index, query_headers_by_head->ne[1]))
    {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    if (!is_scan_query_storage(*query_headers_by_head) ||
        !is_scan_cache_storage(*cache_headers_by_head) ||
        !is_scan_output_storage(*output_by_head) ||
        query_headers_by_head->ne[0] != num_query_tiles ||
        cache_headers_by_head->ne[0] != num_query_tiles ||
        cache_headers_by_head->ne[1] != num_cache_tokens ||
        output_by_head->ne[0] != num_cache_tokens ||
        cache_headers_by_head->ne[2] != query_headers_by_head->ne[1] ||
        output_by_head->ne[1] != query_headers_by_head->ne[1])
    {
        return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
    }

    const openturbo_ggml_tensor_view_t query_head_view = make_scan_query_head_view(*query_headers_by_head, head_index);
    const openturbo_ggml_tensor_view_t cache_head_view = make_scan_cache_head_view(*cache_headers_by_head, head_index);
    const openturbo_ggml_tensor_view_t output_head_view = make_scan_output_head_view(*output_by_head, head_index);
    const openturbo_llama_scan_request_t request{
        query_head_view,
        cache_head_view,
        output_head_view,
        num_query_tiles,
        num_cache_tokens,
        stream_context,
        OPENTURBO_LLAMA_LAYOUT_HEAD_LOCAL_KV_TILES_V1};
    return openturbo_llama_scan(&request, cuda_status_out);
}