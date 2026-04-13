#include "../include/openturbo/c_api.h"
#include "../include/openturbo/ggml_adapter.h"

#include <cuda_runtime.h>

#include <array>
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

        std::cout << "c_api_smoke_test PASSED" << std::endl;
        exit_code = 0;
    } while (false);

    cudaFree(device_headers);
    cudaFree(device_input);
    return exit_code;
}
