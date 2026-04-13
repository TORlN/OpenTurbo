#include "../include/openturbo/c_api.h"
#include "../include/openturbo/ggml_adapter.h"
#include "../include/openturbo/llama_bridge.h"
#include "../include/openturbo/llama_kv_shim.h"
#include "scan_reference.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

#define GGML_MAX_DIMS 4
#define GGML_TYPE_F32 0

enum ggml_type
{
    GGML_TYPE_I8 = 1
};

struct ggml_tensor
{
    void *data;
    int type;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
};

#include "../include/openturbo/ggml_downstream.hpp"

namespace
{
    bool check_status(const char *label, openturbo_status_t status, int cuda_status)
    {
        if (status == OPENTURBO_STATUS_SUCCESS)
        {
            return true;
        }

        std::cerr << label << " failed with status=" << openturbo_status_string(status);
        if (status == OPENTURBO_STATUS_CUDA_ERROR)
        {
            std::cerr << " cuda_status=" << openturbo_cuda_error_string(cuda_status);
        }
        std::cerr << std::endl;
        return false;
    }

    openturbo_ggml_tensor_view_t make_contiguous_1d_view(void *data, uint32_t element_type, int64_t element_count)
    {
        openturbo_ggml_tensor_view_t view{};
        view.data = data;
        view.element_type = element_type;
        view.n_dims = 1;
        view.ne[0] = element_count;
        view.nb[0] = (element_type == OPENTURBO_GGML_TYPE_F32) ? 4u : OPENTURBO_PACKED_TILE_HEADER_BYTES;
        return view;
    }

    openturbo_ggml_tensor_view_t make_contiguous_2d_view(void *data, uint32_t element_type, int64_t dim0, int64_t dim1)
    {
        openturbo_ggml_tensor_view_t view{};
        view.data = data;
        view.element_type = element_type;
        view.n_dims = 2;
        view.ne[0] = dim0;
        view.ne[1] = dim1;
        view.nb[0] = (element_type == OPENTURBO_GGML_TYPE_F32) ? 4u : OPENTURBO_PACKED_TILE_HEADER_BYTES;
        view.nb[1] = view.nb[0] * static_cast<uint64_t>(dim0);
        return view;
    }

    openturbo_ggml_tensor_view_t make_contiguous_3d_view(void *data, uint32_t element_type, int64_t dim0, int64_t dim1, int64_t dim2)
    {
        openturbo_ggml_tensor_view_t view{};
        view.data = data;
        view.element_type = element_type;
        view.n_dims = 3;
        view.ne[0] = dim0;
        view.ne[1] = dim1;
        view.ne[2] = dim2;
        view.nb[0] = (element_type == OPENTURBO_GGML_TYPE_F32) ? 4u : OPENTURBO_PACKED_TILE_HEADER_BYTES;
        view.nb[1] = view.nb[0] * static_cast<uint64_t>(dim0);
        view.nb[2] = view.nb[1] * static_cast<uint64_t>(dim1);
        return view;
    }

    ggml_tensor make_mock_ggml_tensor_2d(void *data, int type, int64_t dim0, int64_t dim1, size_t element_stride)
    {
        ggml_tensor tensor{};
        tensor.data = data;
        tensor.type = type;
        tensor.ne[0] = dim0;
        tensor.ne[1] = dim1;
        tensor.ne[2] = 1;
        tensor.ne[3] = 1;
        tensor.nb[0] = element_stride;
        tensor.nb[1] = tensor.nb[0] * static_cast<size_t>(dim0);
        tensor.nb[2] = tensor.nb[1] * static_cast<size_t>(dim1);
        tensor.nb[3] = tensor.nb[2];
        return tensor;
    }

    ggml_tensor make_mock_ggml_tensor_3d(void *data, int type, int64_t dim0, int64_t dim1, int64_t dim2, size_t element_stride)
    {
        ggml_tensor tensor{};
        tensor.data = data;
        tensor.type = type;
        tensor.ne[0] = dim0;
        tensor.ne[1] = dim1;
        tensor.ne[2] = dim2;
        tensor.ne[3] = 1;
        tensor.nb[0] = element_stride;
        tensor.nb[1] = tensor.nb[0] * static_cast<size_t>(dim0);
        tensor.nb[2] = tensor.nb[1] * static_cast<size_t>(dim1);
        tensor.nb[3] = tensor.nb[2] * static_cast<size_t>(dim2);
        return tensor;
    }

    openturbo_packed_tile_header_t to_c_header(const openturbo::PackedTileHeader &header)
    {
        openturbo_packed_tile_header_t result{};
        std::memcpy(&result, &header, sizeof(result));
        return result;
    }

    bool check_close(const char *label, float actual, float expected, float tolerance = 1.0e-3f)
    {
        if (std::fabs(actual - expected) <= tolerance)
        {
            return true;
        }

        std::cerr << label << " mismatch actual=" << actual << " expected=" << expected << std::endl;
        return false;
    }
}

int main()
{
    if (openturbo_get_c_api_version() != OPENTURBO_CAPI_VERSION)
    {
        std::cerr << "ABI version mismatch" << std::endl;
        return 1;
    }

    std::array<float, OPENTURBO_TILE_DIMS> host_input{};
    for (int index = 0; index < OPENTURBO_TILE_DIMS; ++index)
    {
        host_input[index] = static_cast<float>((index % 19) - 9) * 0.125f;
    }

    float *device_input = nullptr;
    openturbo_packed_tile_header_t *device_headers = nullptr;
    openturbo_packed_tile_header_t host_header{};
    float *device_scan_output = nullptr;

    if (cudaMalloc(&device_input, sizeof(float) * host_input.size()) != cudaSuccess)
    {
        std::cerr << "cudaMalloc(device_input) failed" << std::endl;
        return 1;
    }

    if (cudaMalloc(&device_headers, sizeof(openturbo_packed_tile_header_t)) != cudaSuccess)
    {
        cudaFree(device_input);
        std::cerr << "cudaMalloc(device_headers) failed" << std::endl;
        return 1;
    }

    if (cudaMalloc(&device_scan_output, 4 * sizeof(float)) != cudaSuccess)
    {
        cudaFree(device_headers);
        cudaFree(device_input);
        std::cerr << "cudaMalloc(device_scan_output) failed" << std::endl;
        return 1;
    }

    int exit_code = 1;
    do
    {
        if (cudaMemcpy(device_input, host_input.data(), sizeof(float) * host_input.size(), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            std::cerr << "cudaMemcpy host->device failed" << std::endl;
            break;
        }

        int cuda_status = 0;
        if (!check_status(
                "openturbo_encode_tile_fused",
                openturbo_encode_tile_fused(device_input, device_headers, 1, 13, 10000.0f, nullptr, &cuda_status),
                cuda_status))
        {
            break;
        }

        if (cudaMemcpy(&host_header, device_headers, sizeof(host_header), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::cerr << "cudaMemcpy device->host failed" << std::endl;
            break;
        }

        if (host_header.reserved_u32 != 0u ||
            (host_header.quadrant_word_0 == 0ull && host_header.quadrant_word_1 == 0ull))
        {
            std::cerr << "direct C API encode produced an invalid header" << std::endl;
            break;
        }

        if (cudaMemset(device_headers, 0, sizeof(openturbo_packed_tile_header_t)) != cudaSuccess)
        {
            std::cerr << "cudaMemset failed" << std::endl;
            break;
        }

        const openturbo_ggml_tensor_view_t input_view = make_contiguous_2d_view(device_input, OPENTURBO_GGML_TYPE_F32, OPENTURBO_TILE_DIMS, 1);
        const openturbo_ggml_tensor_view_t output_view = make_contiguous_1d_view(device_headers, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, 1);
        const openturbo_stream_context_t stream_context{nullptr, OPENTURBO_STREAM_CONTEXT_DEFAULT};

        if (!check_status(
                "openturbo_ggml_encode",
                openturbo_ggml_encode(&input_view, &output_view, 13, 10000.0f, stream_context, &cuda_status),
                cuda_status))
        {
            break;
        }

        if (cudaMemcpy(&host_header, device_headers, sizeof(host_header), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::cerr << "cudaMemcpy device->host after ggml encode failed" << std::endl;
            break;
        }

        if (host_header.reserved_u32 != 0u ||
            (host_header.quadrant_word_0 == 0ull && host_header.quadrant_word_1 == 0ull))
        {
            std::cerr << "ggml adapter encode produced an invalid header" << std::endl;
            break;
        }

        std::array<float, OPENTURBO_TILE_DIMS * 2> host_input_by_head{};
        std::memcpy(host_input_by_head.data(), host_input.data(), sizeof(float) * host_input.size());
        std::memcpy(host_input_by_head.data() + OPENTURBO_TILE_DIMS, host_input.data(), sizeof(float) * host_input.size());
        float *device_input_by_head = nullptr;
        openturbo_packed_tile_header_t *device_headers_by_head = nullptr;
        std::array<openturbo_packed_tile_header_t, 2> host_headers_by_head{};
        if (cudaMalloc(&device_input_by_head, sizeof(float) * host_input_by_head.size()) != cudaSuccess)
        {
            std::cerr << "cudaMalloc(device_input_by_head) failed" << std::endl;
            break;
        }
        if (cudaMalloc(&device_headers_by_head, sizeof(openturbo_packed_tile_header_t) * host_headers_by_head.size()) != cudaSuccess)
        {
            cudaFree(device_input_by_head);
            std::cerr << "cudaMalloc(device_headers_by_head) failed" << std::endl;
            break;
        }
        if (cudaMemcpy(device_input_by_head, host_input_by_head.data(), sizeof(float) * host_input_by_head.size(), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cudaFree(device_headers_by_head);
            cudaFree(device_input_by_head);
            std::cerr << "cudaMemcpy multi-head encode input failed" << std::endl;
            break;
        }

        const openturbo_ggml_tensor_view_t input_heads_view = make_contiguous_3d_view(device_input_by_head, OPENTURBO_GGML_TYPE_F32, OPENTURBO_TILE_DIMS, 1, 2);
        const openturbo_ggml_tensor_view_t output_heads_view = make_contiguous_2d_view(device_headers_by_head, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, 1, 2);
        if (!check_status(
                "openturbo_llama_encode_from_kv_heads",
                openturbo_llama_encode_from_kv_heads(&input_heads_view, &output_heads_view, 0, 13, 10000.0f, stream_context, &cuda_status),
                cuda_status))
        {
            cudaFree(device_headers_by_head);
            cudaFree(device_input_by_head);
            break;
        }
        if (cudaMemcpy(host_headers_by_head.data(), device_headers_by_head, sizeof(openturbo_packed_tile_header_t) * host_headers_by_head.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cudaFree(device_headers_by_head);
            cudaFree(device_input_by_head);
            std::cerr << "cudaMemcpy multi-head encode output failed" << std::endl;
            break;
        }
        if (host_headers_by_head[0].reserved_u32 != 0u ||
            (host_headers_by_head[0].quadrant_word_0 == 0ull && host_headers_by_head[0].quadrant_word_1 == 0ull))
        {
            cudaFree(device_headers_by_head);
            cudaFree(device_input_by_head);
            std::cerr << "llama KV shim encode produced an invalid head-local header" << std::endl;
            break;
        }
        if (openturbo_llama_encode_from_kv_heads(&input_heads_view, &output_heads_view, 2, 13, 10000.0f, stream_context, &cuda_status) != OPENTURBO_STATUS_INVALID_ARGUMENT)
        {
            cudaFree(device_headers_by_head);
            cudaFree(device_input_by_head);
            std::cerr << "llama KV shim encode did not reject invalid head index" << std::endl;
            break;
        }

        ggml_tensor ggml_input_heads = make_mock_ggml_tensor_3d(device_input_by_head, GGML_TYPE_F32, OPENTURBO_TILE_DIMS, 1, 2, sizeof(float));
        ggml_tensor ggml_output_heads = make_mock_ggml_tensor_2d(device_headers_by_head, GGML_TYPE_I8, 1, 2, OPENTURBO_PACKED_TILE_HEADER_BYTES);
        if (!check_status(
                "openturbo::ggml_downstream::llama_encode_from_ggml_tensors",
                openturbo::ggml_downstream::llama_encode_from_ggml_tensors(&ggml_input_heads, &ggml_output_heads, 0, 13, 10000.0f, stream_context, &cuda_status),
                cuda_status))
        {
            cudaFree(device_headers_by_head);
            cudaFree(device_input_by_head);
            break;
        }

        ggml_tensor invalid_ggml_input_heads = ggml_input_heads;
        invalid_ggml_input_heads.type = GGML_TYPE_I8;
        if (openturbo::ggml_downstream::llama_encode_from_ggml_tensors(&invalid_ggml_input_heads, &ggml_output_heads, 0, 13, 10000.0f, stream_context, &cuda_status) != OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT)
        {
            cudaFree(device_headers_by_head);
            cudaFree(device_input_by_head);
            std::cerr << "ggml downstream encode did not reject non-f32 input" << std::endl;
            break;
        }

        if (!check_status(
                "openturbo_llama_encode_all_kv_heads",
                openturbo_llama_encode_all_kv_heads(&input_heads_view, &output_heads_view, 13, 10000.0f, stream_context, &cuda_status),
                cuda_status))
        {
            cudaFree(device_headers_by_head);
            cudaFree(device_input_by_head);
            break;
        }
        if (cudaMemcpy(host_headers_by_head.data(), device_headers_by_head, sizeof(openturbo_packed_tile_header_t) * host_headers_by_head.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cudaFree(device_headers_by_head);
            cudaFree(device_input_by_head);
            std::cerr << "cudaMemcpy multi-head encode-all output failed" << std::endl;
            break;
        }
        if (host_headers_by_head[1].reserved_u32 != 0u ||
            (host_headers_by_head[1].quadrant_word_0 == 0ull && host_headers_by_head[1].quadrant_word_1 == 0ull))
        {
            cudaFree(device_headers_by_head);
            cudaFree(device_input_by_head);
            std::cerr << "llama KV shim encode-all did not populate the second head" << std::endl;
            break;
        }

        if (!check_status(
                "openturbo::ggml_downstream::llama_encode_all_heads_from_ggml_tensors",
                openturbo::ggml_downstream::llama_encode_all_heads_from_ggml_tensors(&ggml_input_heads, &ggml_output_heads, 13, 10000.0f, stream_context, &cuda_status),
                cuda_status))
        {
            cudaFree(device_headers_by_head);
            cudaFree(device_input_by_head);
            break;
        }
        cudaFree(device_headers_by_head);
        cudaFree(device_input_by_head);

        std::array<float, OPENTURBO_TILE_DIMS> host_query{};
        std::array<std::array<float, OPENTURBO_TILE_DIMS>, 2> host_cache{};
        std::array<openturbo_packed_tile_header_t, 2> single_tile_cache_headers{};
        std::array<float, 2> host_single_tile_scan{};
        std::array<float, 2> expected_single_tile_scan{};

        for (int index = 0; index < OPENTURBO_TILE_DIMS; ++index)
        {
            host_query[index] = static_cast<float>((index % 13) - 6) * 0.2f;
            host_cache[0][index] = static_cast<float>(((3 * index) % 23) - 11) * 0.125f;
            host_cache[1][index] = static_cast<float>(((5 * index) % 29) - 14) * 0.1f;
        }

        const auto query_fields = openturbo::compute_reference_header(host_query.data(), 7, 10000.0f);
        const auto cache0_fields = openturbo::compute_reference_header(host_cache[0].data(), 3, 10000.0f);
        const auto cache1_fields = openturbo::compute_reference_header(host_cache[1].data(), 9, 10000.0f);
        const auto query_header_cpp = openturbo::pack_reference_header(query_fields);
        const auto cache0_header_cpp = openturbo::pack_reference_header(cache0_fields);
        const auto cache1_header_cpp = openturbo::pack_reference_header(cache1_fields);

        const openturbo_packed_tile_header_t single_tile_query_header = to_c_header(query_header_cpp);
        single_tile_cache_headers[0] = to_c_header(cache0_header_cpp);
        single_tile_cache_headers[1] = to_c_header(cache1_header_cpp);
        expected_single_tile_scan[0] = openturbo::estimate_scan_dot(query_header_cpp, cache0_header_cpp);
        expected_single_tile_scan[1] = openturbo::estimate_scan_dot(query_header_cpp, cache1_header_cpp);

        openturbo_packed_tile_header_t *device_query_header = nullptr;
        openturbo_packed_tile_header_t *device_cache_headers = nullptr;
        if (cudaMalloc(&device_query_header, sizeof(openturbo_packed_tile_header_t)) != cudaSuccess)
        {
            std::cerr << "cudaMalloc(device_query_header) failed" << std::endl;
            break;
        }
        if (cudaMalloc(&device_cache_headers, sizeof(openturbo_packed_tile_header_t) * single_tile_cache_headers.size()) != cudaSuccess)
        {
            cudaFree(device_query_header);
            std::cerr << "cudaMalloc(device_cache_headers) failed" << std::endl;
            break;
        }

        if (cudaMemcpy(device_query_header, &single_tile_query_header, sizeof(single_tile_query_header), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(device_cache_headers, single_tile_cache_headers.data(), sizeof(openturbo_packed_tile_header_t) * single_tile_cache_headers.size(), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "cudaMemcpy single-tile headers failed" << std::endl;
            break;
        }

        if (!check_status(
                "openturbo_scan_query_many_cache",
                openturbo_scan_query_many_cache(device_query_header, device_cache_headers, device_scan_output, static_cast<int>(single_tile_cache_headers.size()), nullptr, &cuda_status),
                cuda_status))
        {
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }
        if (cudaMemcpy(host_single_tile_scan.data(), device_scan_output, sizeof(float) * host_single_tile_scan.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "cudaMemcpy single-tile scan output failed" << std::endl;
            break;
        }
        if (!check_close("direct single-tile scan[0]", host_single_tile_scan[0], expected_single_tile_scan[0]) ||
            !check_close("direct single-tile scan[1]", host_single_tile_scan[1], expected_single_tile_scan[1]))
        {
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }

        const openturbo_ggml_tensor_view_t query_view = make_contiguous_1d_view(device_query_header, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, 1);
        const openturbo_ggml_tensor_view_t cache_view = make_contiguous_1d_view(device_cache_headers, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, static_cast<int64_t>(single_tile_cache_headers.size()));
        const openturbo_ggml_tensor_view_t output_scan_view = make_contiguous_1d_view(device_scan_output, OPENTURBO_GGML_TYPE_F32, static_cast<int64_t>(host_single_tile_scan.size()));

        if (!check_status(
                "openturbo_ggml_scan_query_many_cache",
                openturbo_ggml_scan_query_many_cache(&query_view, &cache_view, &output_scan_view, stream_context, &cuda_status),
                cuda_status))
        {
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }
        if (cudaMemcpy(host_single_tile_scan.data(), device_scan_output, sizeof(float) * host_single_tile_scan.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "cudaMemcpy ggml single-tile scan output failed" << std::endl;
            break;
        }
        if (!check_close("ggml single-tile scan[0]", host_single_tile_scan[0], expected_single_tile_scan[0]) ||
            !check_close("ggml single-tile scan[1]", host_single_tile_scan[1], expected_single_tile_scan[1]))
        {
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }

        std::array<std::array<float, OPENTURBO_TILE_DIMS>, 2> host_query_tiles{};
        std::array<std::array<std::array<float, OPENTURBO_TILE_DIMS>, 2>, 2> host_cache_tiles{};
        std::array<openturbo_packed_tile_header_t, 2> multi_query_headers{};
        std::array<openturbo_packed_tile_header_t, 4> multi_cache_headers{};
        std::array<float, 2> host_multi_tile_scan{};
        std::array<float, 2> expected_multi_tile_scan{};

        for (int tile = 0; tile < 2; ++tile)
        {
            for (int index = 0; index < OPENTURBO_TILE_DIMS; ++index)
            {
                host_query_tiles[tile][index] = static_cast<float>(((tile + 2) * index % 31) - 15) * 0.08f;
                host_cache_tiles[0][tile][index] = static_cast<float>(((tile + 3) * index % 17) - 8) * 0.11f;
                host_cache_tiles[1][tile][index] = static_cast<float>(((tile + 5) * index % 19) - 9) * 0.09f;
            }

            const auto multi_query_fields = openturbo::compute_reference_header(host_query_tiles[tile].data(), 5 + tile, 10000.0f);
            multi_query_headers[tile] = to_c_header(openturbo::pack_reference_header(multi_query_fields));

            const auto cache_token0_fields = openturbo::compute_reference_header(host_cache_tiles[0][tile].data(), 11 + tile, 10000.0f);
            const auto cache_token1_fields = openturbo::compute_reference_header(host_cache_tiles[1][tile].data(), 15 + tile, 10000.0f);
            multi_cache_headers[tile] = to_c_header(openturbo::pack_reference_header(cache_token0_fields));
            multi_cache_headers[2 + tile] = to_c_header(openturbo::pack_reference_header(cache_token1_fields));
        }

        const auto *query_headers_cpp = reinterpret_cast<const openturbo::PackedTileHeader *>(multi_query_headers.data());
        const auto *cache_headers_cpp = reinterpret_cast<const openturbo::PackedTileHeader *>(multi_cache_headers.data());
        expected_multi_tile_scan[0] = openturbo::estimate_scan_dot_multi_tile(query_headers_cpp, cache_headers_cpp, 2);
        expected_multi_tile_scan[1] = openturbo::estimate_scan_dot_multi_tile(query_headers_cpp, cache_headers_cpp + 2, 2);

        openturbo_packed_tile_header_t *device_multi_query_headers = nullptr;
        openturbo_packed_tile_header_t *device_multi_cache_headers = nullptr;
        if (cudaMalloc(&device_multi_query_headers, sizeof(openturbo_packed_tile_header_t) * multi_query_headers.size()) != cudaSuccess)
        {
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "cudaMalloc(device_multi_query_headers) failed" << std::endl;
            break;
        }
        if (cudaMalloc(&device_multi_cache_headers, sizeof(openturbo_packed_tile_header_t) * multi_cache_headers.size()) != cudaSuccess)
        {
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "cudaMalloc(device_multi_cache_headers) failed" << std::endl;
            break;
        }

        if (cudaMemcpy(device_multi_query_headers, multi_query_headers.data(), sizeof(openturbo_packed_tile_header_t) * multi_query_headers.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(device_multi_cache_headers, multi_cache_headers.data(), sizeof(openturbo_packed_tile_header_t) * multi_cache_headers.size(), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "cudaMemcpy multi-tile headers failed" << std::endl;
            break;
        }

        if (!check_status(
                "openturbo_scan_query_many_cache_multi_tile",
                openturbo_scan_query_many_cache_multi_tile(device_multi_query_headers, device_multi_cache_headers, device_scan_output, 2, 2, nullptr, &cuda_status),
                cuda_status))
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }
        if (cudaMemcpy(host_multi_tile_scan.data(), device_scan_output, sizeof(float) * host_multi_tile_scan.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "cudaMemcpy multi-tile scan output failed" << std::endl;
            break;
        }
        if (!check_close("direct multi-tile scan[0]", host_multi_tile_scan[0], expected_multi_tile_scan[0]) ||
            !check_close("direct multi-tile scan[1]", host_multi_tile_scan[1], expected_multi_tile_scan[1]))
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }

        const openturbo_ggml_tensor_view_t multi_query_view = make_contiguous_1d_view(device_multi_query_headers, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, static_cast<int64_t>(multi_query_headers.size()));
        const openturbo_ggml_tensor_view_t multi_cache_view = make_contiguous_2d_view(device_multi_cache_headers, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, 2, 2);
        const openturbo_ggml_tensor_view_t multi_output_view = make_contiguous_1d_view(device_scan_output, OPENTURBO_GGML_TYPE_F32, static_cast<int64_t>(host_multi_tile_scan.size()));

        openturbo_ggml_tensor_view_t invalid_encode_input_view = input_view;
        invalid_encode_input_view.n_dims = 1;
        if (openturbo_ggml_encode(&invalid_encode_input_view, &output_view, 13, 10000.0f, stream_context, &cuda_status) != OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT)
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "ggml encode validation did not reject invalid rank" << std::endl;
            break;
        }

        openturbo_ggml_tensor_view_t invalid_encode_stride_view = input_view;
        invalid_encode_stride_view.nb[1] += 4;
        if (openturbo_ggml_encode(&invalid_encode_stride_view, &output_view, 13, 10000.0f, stream_context, &cuda_status) != OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT)
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "ggml encode validation did not reject invalid stride" << std::endl;
            break;
        }

        if (!check_status(
                "openturbo_ggml_scan_query_many_cache_multi_tile",
                openturbo_ggml_scan_query_many_cache_multi_tile(&multi_query_view, &multi_cache_view, &multi_output_view, 2, 2, stream_context, &cuda_status),
                cuda_status))
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }
        if (cudaMemcpy(host_multi_tile_scan.data(), device_scan_output, sizeof(float) * host_multi_tile_scan.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "cudaMemcpy ggml multi-tile scan output failed" << std::endl;
            break;
        }
        if (!check_close("ggml multi-tile scan[0]", host_multi_tile_scan[0], expected_multi_tile_scan[0]) ||
            !check_close("ggml multi-tile scan[1]", host_multi_tile_scan[1], expected_multi_tile_scan[1]))
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }

        openturbo_ggml_tensor_view_t invalid_query_view = query_view;
        invalid_query_view.ne[0] = 2;
        if (openturbo_ggml_scan_query_many_cache(&invalid_query_view, &cache_view, &output_scan_view, stream_context, &cuda_status) != OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT)
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "ggml single-tile layout validation did not reject invalid query shape" << std::endl;
            break;
        }

        openturbo_ggml_tensor_view_t invalid_output_view = output_scan_view;
        invalid_output_view.ne[0] = 1;
        if (openturbo_ggml_scan_query_many_cache(&query_view, &cache_view, &invalid_output_view, stream_context, &cuda_status) != OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT)
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "ggml single-tile layout validation did not reject undersized output" << std::endl;
            break;
        }

        openturbo_ggml_tensor_view_t invalid_multi_cache_view = multi_cache_view;
        invalid_multi_cache_view.ne[0] = 3;
        if (openturbo_ggml_scan_query_many_cache_multi_tile(&multi_query_view, &invalid_multi_cache_view, &multi_output_view, 2, 2, stream_context, &cuda_status) != OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT)
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "ggml multi-tile layout validation did not reject invalid cache count" << std::endl;
            break;
        }

        openturbo_ggml_tensor_view_t invalid_multi_cache_stride_view = multi_cache_view;
        invalid_multi_cache_stride_view.nb[1] += OPENTURBO_PACKED_TILE_HEADER_BYTES;
        if (openturbo_ggml_scan_query_many_cache_multi_tile(&multi_query_view, &invalid_multi_cache_stride_view, &multi_output_view, 2, 2, stream_context, &cuda_status) != OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT)
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "ggml multi-tile layout validation did not reject invalid cache stride" << std::endl;
            break;
        }

        const openturbo_llama_encode_request_t llama_encode_request{
            input_view,
            output_view,
            13,
            10000.0f,
            stream_context,
            OPENTURBO_LLAMA_LAYOUT_HEAD_LOCAL_KV_TILES_V1};
        if (!check_status(
                "openturbo_llama_encode",
                openturbo_llama_encode(&llama_encode_request, &cuda_status),
                cuda_status))
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }

        const openturbo_llama_scan_request_t llama_single_scan_request{
            query_view,
            cache_view,
            output_scan_view,
            1,
            static_cast<int>(single_tile_cache_headers.size()),
            stream_context,
            OPENTURBO_LLAMA_LAYOUT_HEAD_LOCAL_KV_TILES_V1};
        if (!check_status(
                "openturbo_llama_scan(single)",
                openturbo_llama_scan(&llama_single_scan_request, &cuda_status),
                cuda_status))
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }

        const openturbo_llama_scan_request_t llama_multi_scan_request{
            multi_query_view,
            multi_cache_view,
            multi_output_view,
            2,
            2,
            stream_context,
            OPENTURBO_LLAMA_LAYOUT_HEAD_LOCAL_KV_TILES_V1};
        if (!check_status(
                "openturbo_llama_scan(multi)",
                openturbo_llama_scan(&llama_multi_scan_request, &cuda_status),
                cuda_status))
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }

        openturbo_llama_encode_request_t invalid_llama_encode_request = llama_encode_request;
        invalid_llama_encode_request.layout = 99u;
        if (openturbo_llama_encode(&invalid_llama_encode_request, &cuda_status) != OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT)
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "llama bridge encode path did not reject unsupported layout id" << std::endl;
            break;
        }

        openturbo_llama_scan_request_t invalid_llama_request = llama_single_scan_request;
        invalid_llama_request.layout = 99u;
        if (openturbo_llama_scan(&invalid_llama_request, &cuda_status) != OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT)
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "llama bridge scan path did not reject unsupported layout id" << std::endl;
            break;
        }

        openturbo_llama_scan_request_t invalid_llama_count_request = llama_multi_scan_request;
        invalid_llama_count_request.num_cache_tokens = 3;
        if (openturbo_llama_scan(&invalid_llama_count_request, &cuda_status) != OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT)
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "llama bridge did not reject mismatched cache token count" << std::endl;
            break;
        }

        std::array<openturbo_packed_tile_header_t, 4> multi_query_headers_by_head{};
        std::array<openturbo_packed_tile_header_t, 8> multi_cache_headers_by_head{};
        std::array<float, 4> host_multi_head_scan_output{};
        for (int tile = 0; tile < 2; ++tile)
        {
            multi_query_headers_by_head[tile] = multi_query_headers[tile];
            multi_query_headers_by_head[2 + tile] = multi_query_headers[tile];
            multi_cache_headers_by_head[tile] = multi_cache_headers[tile];
            multi_cache_headers_by_head[2 + tile] = multi_cache_headers[2 + tile];
            multi_cache_headers_by_head[4 + tile] = multi_cache_headers[tile];
            multi_cache_headers_by_head[6 + tile] = multi_cache_headers[2 + tile];
        }

        openturbo_packed_tile_header_t *device_multi_query_headers_by_head = nullptr;
        openturbo_packed_tile_header_t *device_multi_cache_headers_by_head = nullptr;
        float *device_multi_head_scan_output = nullptr;
        if (cudaMalloc(&device_multi_query_headers_by_head, sizeof(openturbo_packed_tile_header_t) * multi_query_headers_by_head.size()) != cudaSuccess)
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "cudaMalloc(device_multi_query_headers_by_head) failed" << std::endl;
            break;
        }
        if (cudaMalloc(&device_multi_cache_headers_by_head, sizeof(openturbo_packed_tile_header_t) * multi_cache_headers_by_head.size()) != cudaSuccess)
        {
            cudaFree(device_multi_query_headers_by_head);
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "cudaMalloc(device_multi_cache_headers_by_head) failed" << std::endl;
            break;
        }
        if (cudaMalloc(&device_multi_head_scan_output, sizeof(float) * host_multi_head_scan_output.size()) != cudaSuccess)
        {
            cudaFree(device_multi_cache_headers_by_head);
            cudaFree(device_multi_query_headers_by_head);
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "cudaMalloc(device_multi_head_scan_output) failed" << std::endl;
            break;
        }
        if (cudaMemcpy(device_multi_query_headers_by_head, multi_query_headers_by_head.data(), sizeof(openturbo_packed_tile_header_t) * multi_query_headers_by_head.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(device_multi_cache_headers_by_head, multi_cache_headers_by_head.data(), sizeof(openturbo_packed_tile_header_t) * multi_cache_headers_by_head.size(), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cudaFree(device_multi_head_scan_output);
            cudaFree(device_multi_cache_headers_by_head);
            cudaFree(device_multi_query_headers_by_head);
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "cudaMemcpy multi-head scan headers failed" << std::endl;
            break;
        }

        const openturbo_ggml_tensor_view_t query_heads_view = make_contiguous_2d_view(device_multi_query_headers_by_head, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, 2, 2);
        const openturbo_ggml_tensor_view_t cache_heads_view = make_contiguous_3d_view(device_multi_cache_headers_by_head, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, 2, 2, 2);
        const openturbo_ggml_tensor_view_t output_heads_scan_view = make_contiguous_2d_view(device_multi_head_scan_output, OPENTURBO_GGML_TYPE_F32, 2, 2);
        if (!check_status(
                "openturbo_llama_scan_from_kv_cache",
                openturbo_llama_scan_from_kv_cache(&query_heads_view, &cache_heads_view, &output_heads_scan_view, 0, 2, 2, stream_context, &cuda_status),
                cuda_status))
        {
            cudaFree(device_multi_head_scan_output);
            cudaFree(device_multi_cache_headers_by_head);
            cudaFree(device_multi_query_headers_by_head);
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }
        if (cudaMemcpy(host_multi_head_scan_output.data(), device_multi_head_scan_output, sizeof(float) * host_multi_head_scan_output.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cudaFree(device_multi_head_scan_output);
            cudaFree(device_multi_cache_headers_by_head);
            cudaFree(device_multi_query_headers_by_head);
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "cudaMemcpy multi-head scan output failed" << std::endl;
            break;
        }
        if (!check_close("llama KV shim multi-head scan[0]", host_multi_head_scan_output[0], expected_multi_tile_scan[0]) ||
            !check_close("llama KV shim multi-head scan[1]", host_multi_head_scan_output[1], expected_multi_tile_scan[1]))
        {
            cudaFree(device_multi_head_scan_output);
            cudaFree(device_multi_cache_headers_by_head);
            cudaFree(device_multi_query_headers_by_head);
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }
        if (openturbo_llama_scan_from_kv_cache(&query_heads_view, &cache_heads_view, &output_heads_scan_view, 2, 2, 2, stream_context, &cuda_status) != OPENTURBO_STATUS_INVALID_ARGUMENT)
        {
            cudaFree(device_multi_head_scan_output);
            cudaFree(device_multi_cache_headers_by_head);
            cudaFree(device_multi_query_headers_by_head);
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "llama KV shim scan did not reject invalid head index" << std::endl;
            break;
        }

        ggml_tensor ggml_query_heads = make_mock_ggml_tensor_2d(device_multi_query_headers_by_head, GGML_TYPE_I8, 2, 2, OPENTURBO_PACKED_TILE_HEADER_BYTES);
        ggml_tensor ggml_cache_heads = make_mock_ggml_tensor_3d(device_multi_cache_headers_by_head, GGML_TYPE_I8, 2, 2, 2, OPENTURBO_PACKED_TILE_HEADER_BYTES);
        ggml_tensor ggml_scan_output_heads = make_mock_ggml_tensor_2d(device_multi_head_scan_output, GGML_TYPE_F32, 2, 2, sizeof(float));
        if (!check_status(
                "openturbo::ggml_downstream::llama_scan_from_ggml_tensors",
                openturbo::ggml_downstream::llama_scan_from_ggml_tensors(&ggml_query_heads, &ggml_cache_heads, &ggml_scan_output_heads, 0, 2, 2, stream_context, &cuda_status),
                cuda_status))
        {
            cudaFree(device_multi_head_scan_output);
            cudaFree(device_multi_cache_headers_by_head);
            cudaFree(device_multi_query_headers_by_head);
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }

        ggml_tensor invalid_ggml_output_heads = ggml_scan_output_heads;
        invalid_ggml_output_heads.type = GGML_TYPE_I8;
        if (openturbo::ggml_downstream::llama_scan_from_ggml_tensors(&ggml_query_heads, &ggml_cache_heads, &invalid_ggml_output_heads, 0, 2, 2, stream_context, &cuda_status) != OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT)
        {
            cudaFree(device_multi_head_scan_output);
            cudaFree(device_multi_cache_headers_by_head);
            cudaFree(device_multi_query_headers_by_head);
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "ggml downstream scan did not reject non-f32 output" << std::endl;
            break;
        }

        if (!check_status(
                "openturbo_llama_scan_all_kv_heads",
                openturbo_llama_scan_all_kv_heads(&query_heads_view, &cache_heads_view, &output_heads_scan_view, 2, 2, stream_context, &cuda_status),
                cuda_status))
        {
            cudaFree(device_multi_head_scan_output);
            cudaFree(device_multi_cache_headers_by_head);
            cudaFree(device_multi_query_headers_by_head);
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }
        if (cudaMemcpy(host_multi_head_scan_output.data(), device_multi_head_scan_output, sizeof(float) * host_multi_head_scan_output.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cudaFree(device_multi_head_scan_output);
            cudaFree(device_multi_cache_headers_by_head);
            cudaFree(device_multi_query_headers_by_head);
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "cudaMemcpy multi-head scan-all output failed" << std::endl;
            break;
        }
        if (!check_close("llama KV shim scan-all head1[0]", host_multi_head_scan_output[2], expected_multi_tile_scan[0]) ||
            !check_close("llama KV shim scan-all head1[1]", host_multi_head_scan_output[3], expected_multi_tile_scan[1]))
        {
            cudaFree(device_multi_head_scan_output);
            cudaFree(device_multi_cache_headers_by_head);
            cudaFree(device_multi_query_headers_by_head);
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }

        if (!check_status(
                "openturbo::ggml_downstream::llama_scan_all_heads_from_ggml_tensors",
                openturbo::ggml_downstream::llama_scan_all_heads_from_ggml_tensors(&ggml_query_heads, &ggml_cache_heads, &ggml_scan_output_heads, 2, 2, stream_context, &cuda_status),
                cuda_status))
        {
            cudaFree(device_multi_head_scan_output);
            cudaFree(device_multi_cache_headers_by_head);
            cudaFree(device_multi_query_headers_by_head);
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            break;
        }

        cudaFree(device_multi_head_scan_output);
        cudaFree(device_multi_cache_headers_by_head);
        cudaFree(device_multi_query_headers_by_head);

        cudaFree(device_multi_cache_headers);
        cudaFree(device_multi_query_headers);
        cudaFree(device_cache_headers);
        cudaFree(device_query_header);

        std::cout << "c_api_smoke_test PASSED" << std::endl;
        exit_code = 0;
    } while (false);

    cudaFree(device_scan_output);
    cudaFree(device_headers);
    cudaFree(device_input);
    return exit_code;
}
