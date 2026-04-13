#ifndef OPENTURBO_CAPI_EXPORTS
#define OPENTURBO_CAPI_EXPORTS
#endif

#include "../include/openturbo/ggml_adapter.h"

#include <stddef.h>

namespace
{
    uint64_t element_count(const openturbo_ggml_tensor_view_t &view)
    {
        uint64_t count = 1;
        for (int dim = 0; dim < view.n_dims; ++dim)
        {
            if (view.ne[dim] <= 0)
            {
                return 0;
            }
            count *= static_cast<uint64_t>(view.ne[dim]);
        }
        return count;
    }

    uint64_t contiguous_stride_bytes(uint32_t element_type)
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

    bool is_contiguous(const openturbo_ggml_tensor_view_t &view)
    {
        if (view.data == nullptr || view.n_dims <= 0 || view.n_dims > OPENTURBO_GGML_MAX_DIMS)
        {
            return false;
        }

        uint64_t expected_stride = contiguous_stride_bytes(view.element_type);
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

    void *normalize_stream_context(openturbo_stream_context_t stream_context)
    {
        return stream_context.cuda_stream;
    }

    bool has_exact_element_count(const openturbo_ggml_tensor_view_t &view, uint64_t expected_count)
    {
        return element_count(view) == expected_count;
    }
}

extern "C" OPENTURBO_CAPI openturbo_status_t openturbo_ggml_encode(
    const openturbo_ggml_tensor_view_t *input,
    const openturbo_ggml_tensor_view_t *output_headers,
    int token_pos,
    float rope_theta,
    openturbo_stream_context_t stream_context,
    int *cuda_status_out)
{
    if (input == nullptr || output_headers == nullptr)
    {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    if (input->element_type != OPENTURBO_GGML_TYPE_F32 ||
        output_headers->element_type != OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER ||
        !is_contiguous(*input) || !is_contiguous(*output_headers))
    {
        return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
    }

    const uint64_t input_elements = element_count(*input);
    if (input_elements == 0 || (input_elements % OPENTURBO_TILE_DIMS) != 0)
    {
        return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
    }

    const uint64_t num_tiles = input_elements / OPENTURBO_TILE_DIMS;
    if (!has_exact_element_count(*output_headers, num_tiles))
    {
        return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
    }

    return openturbo_encode_tile_fused(
        static_cast<const float *>(input->data),
        static_cast<openturbo_packed_tile_header_t *>(output_headers->data),
        static_cast<int>(num_tiles),
        token_pos,
        rope_theta,
        normalize_stream_context(stream_context),
        cuda_status_out);
}

extern "C" OPENTURBO_CAPI openturbo_status_t openturbo_ggml_scan_query_many_cache(
    const openturbo_ggml_tensor_view_t *query_header,
    const openturbo_ggml_tensor_view_t *cache_headers,
    const openturbo_ggml_tensor_view_t *output,
    openturbo_stream_context_t stream_context,
    int *cuda_status_out)
{
    if (query_header == nullptr || cache_headers == nullptr || output == nullptr)
    {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    if (query_header->element_type != OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER ||
        cache_headers->element_type != OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER ||
        output->element_type != OPENTURBO_GGML_TYPE_F32 ||
        !is_contiguous(*query_header) ||
        !is_contiguous(*cache_headers) ||
        !is_contiguous(*output))
    {
        return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
    }

    const uint64_t cache_count = element_count(*cache_headers);
    if (!has_exact_element_count(*query_header, 1) ||
        cache_count == 0 ||
        !has_exact_element_count(*output, cache_count))
    {
        return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
    }

    return openturbo_scan_query_many_cache(
        static_cast<const openturbo_packed_tile_header_t *>(query_header->data),
        static_cast<const openturbo_packed_tile_header_t *>(cache_headers->data),
        static_cast<float *>(output->data),
        static_cast<int>(cache_count),
        normalize_stream_context(stream_context),
        cuda_status_out);
}

extern "C" OPENTURBO_CAPI openturbo_status_t openturbo_ggml_scan_query_many_cache_multi_tile(
    const openturbo_ggml_tensor_view_t *query_headers,
    const openturbo_ggml_tensor_view_t *cache_headers,
    const openturbo_ggml_tensor_view_t *output,
    int num_query_tiles,
    int num_cache_tokens,
    openturbo_stream_context_t stream_context,
    int *cuda_status_out)
{
    if (query_headers == nullptr || cache_headers == nullptr || output == nullptr || num_query_tiles <= 0 || num_cache_tokens <= 0)
    {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    if (query_headers->element_type != OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER ||
        cache_headers->element_type != OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER ||
        output->element_type != OPENTURBO_GGML_TYPE_F32 ||
        !is_contiguous(*query_headers) ||
        !is_contiguous(*cache_headers) ||
        !is_contiguous(*output))
    {
        return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
    }

    if (!has_exact_element_count(*query_headers, static_cast<uint64_t>(num_query_tiles)) ||
        !has_exact_element_count(*cache_headers, static_cast<uint64_t>(num_query_tiles) * static_cast<uint64_t>(num_cache_tokens)) ||
        !has_exact_element_count(*output, static_cast<uint64_t>(num_cache_tokens)))
    {
        return OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT;
    }

    return openturbo_scan_query_many_cache_multi_tile(
        static_cast<const openturbo_packed_tile_header_t *>(query_headers->data),
        static_cast<const openturbo_packed_tile_header_t *>(cache_headers->data),
        static_cast<float *>(output->data),
        num_query_tiles,
        num_cache_tokens,
        normalize_stream_context(stream_context),
        cuda_status_out);
}
