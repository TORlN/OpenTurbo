#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

#include <cstdio>
#include <cstdlib>

namespace openturbo
{
    struct alignas(32) PackedTileHeader
    {
        uint64_t quadrant_word_0;
        uint64_t quadrant_word_1;
        uint64_t qjl_sign_word;
        __half block_scale_fp16;
        __half local_alpha_fp16;
        uint32_t reserved_u32;
    };

    __global__ void encode_tile_fused_kernel(
        const float *__restrict__ input,
        PackedTileHeader *__restrict__ output_headers,
        int num_tiles,
        int token_pos,
        float rope_theta);
}

namespace
{
    constexpr int kTileDims = 128;
    constexpr int kThreadsPerBlock = 32;

    void check_cuda(cudaError_t status, const char *what)
    {
        if (status == cudaSuccess)
        {
            return;
        }

        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }

    void fill_test_tile(float *values, int count)
    {
        for (int i = 0; i < count; ++i)
        {
            const float sign = (i & 1) ? -1.0f : 1.0f;
            values[i] = sign * (0.05f * static_cast<float>(i + 1));
        }
    }
}

int main()
{
    float host_input[kTileDims];
    fill_test_tile(host_input, kTileDims);

    float *device_input = nullptr;
    openturbo::PackedTileHeader *device_header = nullptr;
    openturbo::PackedTileHeader host_header{};

    check_cuda(cudaMalloc(&device_input, sizeof(host_input)), "cudaMalloc(device_input)");
    check_cuda(cudaMalloc(&device_header, sizeof(openturbo::PackedTileHeader)), "cudaMalloc(device_header)");

    check_cuda(
        cudaMemcpy(device_input, host_input, sizeof(host_input), cudaMemcpyHostToDevice),
        "cudaMemcpy(host_input -> device_input)");
    check_cuda(
        cudaMemset(device_header, 0, sizeof(openturbo::PackedTileHeader)),
        "cudaMemset(device_header)");

    openturbo::encode_tile_fused_kernel<<<1, kThreadsPerBlock>>>(
        device_input,
        device_header,
        1,
        0,
        10000.0f);

    check_cuda(cudaGetLastError(), "encode_tile_fused_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check_cuda(
        cudaMemcpy(&host_header, device_header, sizeof(host_header), cudaMemcpyDeviceToHost),
        "cudaMemcpy(device_header -> host_header)");

    std::printf("quadrant_word_0: 0x%016llx\n", static_cast<unsigned long long>(host_header.quadrant_word_0));
    std::printf("quadrant_word_1: 0x%016llx\n", static_cast<unsigned long long>(host_header.quadrant_word_1));
    std::printf("qjl_sign_word:  0x%016llx\n", static_cast<unsigned long long>(host_header.qjl_sign_word));
    std::printf("block_scale:    %f\n", __half2float(host_header.block_scale_fp16));
    std::printf("local_alpha:    %f\n", __half2float(host_header.local_alpha_fp16));
    std::printf("reserved_u32:   0x%08x\n", host_header.reserved_u32);

    check_cuda(cudaFree(device_header), "cudaFree(device_header)");
    check_cuda(cudaFree(device_input), "cudaFree(device_input)");

    return 0;
}