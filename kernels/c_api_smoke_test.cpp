#include "../include/openturbo/c_api.h"
#include "../include/openturbo/ggml_adapter.h"
#include "../include/openturbo/llama_bridge.h"
#include "scan_reference.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

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

    openturbo_ggml_tensor_view_t make_contiguous_view(void *data, uint32_t element_type, int64_t element_count)
    {
        openturbo_ggml_tensor_view_t view{};
        view.data = data;
        view.element_type = element_type;
        view.n_dims = 1;
        view.ne[0] = element_count;
        view.nb[0] = (element_type == OPENTURBO_GGML_TYPE_F32) ? 4u : OPENTURBO_PACKED_TILE_HEADER_BYTES;
        return view;
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

        const openturbo_ggml_tensor_view_t input_view = make_contiguous_view(device_input, OPENTURBO_GGML_TYPE_F32, OPENTURBO_TILE_DIMS);
        const openturbo_ggml_tensor_view_t output_view = make_contiguous_view(device_headers, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, 1);
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

        const openturbo_ggml_tensor_view_t query_view = make_contiguous_view(device_query_header, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, 1);
        const openturbo_ggml_tensor_view_t cache_view = make_contiguous_view(device_cache_headers, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, static_cast<int64_t>(single_tile_cache_headers.size()));
        const openturbo_ggml_tensor_view_t output_scan_view = make_contiguous_view(device_scan_output, OPENTURBO_GGML_TYPE_F32, static_cast<int64_t>(host_single_tile_scan.size()));

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

        const openturbo_ggml_tensor_view_t multi_query_view = make_contiguous_view(device_multi_query_headers, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, static_cast<int64_t>(multi_query_headers.size()));
        const openturbo_ggml_tensor_view_t multi_cache_view = make_contiguous_view(device_multi_cache_headers, OPENTURBO_GGML_TYPE_PACKED_TILE_HEADER, static_cast<int64_t>(multi_cache_headers.size()));
        const openturbo_ggml_tensor_view_t multi_output_view = make_contiguous_view(device_scan_output, OPENTURBO_GGML_TYPE_F32, static_cast<int64_t>(host_multi_tile_scan.size()));

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

        const openturbo_llama_encode_request_t llama_encode_request{
            input_view,
            output_view,
            13,
            10000.0f,
            stream_context,
            OPENTURBO_LLAMA_LAYOUT_FLAT_TILES};
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
            OPENTURBO_LLAMA_LAYOUT_FLAT_TILES};
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
            OPENTURBO_LLAMA_LAYOUT_FLAT_TILES};
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

        openturbo_llama_scan_request_t invalid_llama_request = llama_single_scan_request;
        invalid_llama_request.layout = 99u;
        if (openturbo_llama_scan(&invalid_llama_request, &cuda_status) != OPENTURBO_STATUS_INCOMPATIBLE_LAYOUT)
        {
            cudaFree(device_multi_cache_headers);
            cudaFree(device_multi_query_headers);
            cudaFree(device_cache_headers);
            cudaFree(device_query_header);
            std::cerr << "llama bridge did not reject unsupported layout id" << std::endl;
            break;
        }

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
