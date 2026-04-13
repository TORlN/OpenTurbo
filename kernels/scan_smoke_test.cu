#include "encoder_layout.cuh"
#include "encoder_reference.hpp"
#include "scan_reference.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace openturbo
{
    __global__ void scan_tile_dot_kernel(
        const PackedTileHeader *__restrict__ query_headers,
        const PackedTileHeader *__restrict__ cache_headers,
        float *__restrict__ output,
        int num_pairs);
}

namespace
{
    constexpr float kScanAbsTolerance = 1e-2f;
    constexpr float kScanRelTolerance = 1e-5f;

    struct TileCase
    {
        const char *name;
        openturbo::PackedTileHeader header;
    };

    struct ScanPairCase
    {
        const char *query_name;
        const char *cache_name;
        openturbo::PackedTileHeader query_header;
        openturbo::PackedTileHeader cache_header;
        float expected;
    };

    void check_cuda(cudaError_t status, const char *what)
    {
        if (status == cudaSuccess)
        {
            return;
        }

        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }

    void require_close(float actual, float expected, const char *message)
    {
        const float abs_error = std::fabs(actual - expected);
        const float tolerance = fmaxf(kScanAbsTolerance, kScanRelTolerance * std::fabs(expected));
        if (abs_error <= tolerance)
        {
            return;
        }

        std::fprintf(
            stderr,
            "ASSERTION FAILED: %s (actual=%f expected=%f abs_error=%f tolerance=%f)\n",
            message,
            actual,
            expected,
            abs_error,
            tolerance);
        std::exit(1);
    }

    void fill_alternating_tile(float *values, int count)
    {
        for (int i = 0; i < count; ++i)
        {
            const float sign = (i & 1) ? -1.0f : 1.0f;
            values[i] = sign * (0.05f * static_cast<float>(i + 1));
        }
    }

    void fill_seeded_random_tile(float *values, int count)
    {
        uint32_t state = 0x12345678u;
        for (int i = 0; i < count; ++i)
        {
            state = 1664525u * state + 1013904223u;
            const float unit = static_cast<float>(state & 0x00ffffffu) / static_cast<float>(0x01000000u);
            values[i] = (2.0f * unit - 1.0f) * 3.0f;
        }
    }

    void fill_sparse_spike_tile(float *values, int count)
    {
        for (int i = 0; i < count; ++i)
        {
            values[i] = ((i % 5) == 0) ? 0.0005f * static_cast<float>(i + 1) : 0.0f;
        }

        values[3] = 14.0f;
        values[17] = -11.5f;
        values[71] = 9.25f;
        values[95] = -7.75f;
        values[126] = 5.5f;
    }

    TileCase make_case(const char *name, void (*fill_tile)(float *, int))
    {
        float input[openturbo::kTileDims];
        fill_tile(input, openturbo::kTileDims);
        const openturbo::ReferenceHeaderFields reference = openturbo::compute_reference_header(input, 0, 10000.0f);

        TileCase tile_case{};
        tile_case.name = name;
        tile_case.header = openturbo::pack_reference_header(reference);
        return tile_case;
    }
}

int main()
{
    const TileCase alternating = make_case("Alternating Ramp Tile", fill_alternating_tile);
    const TileCase random = make_case("Seeded Random Tile", fill_seeded_random_tile);
    const TileCase sparse = make_case("Sparse Spike Tile", fill_sparse_spike_tile);

    ScanPairCase pair_cases[] = {
        {alternating.name, alternating.name, alternating.header, alternating.header, openturbo::estimate_scan_dot(alternating.header, alternating.header)},
        {alternating.name, random.name, alternating.header, random.header, openturbo::estimate_scan_dot(alternating.header, random.header)},
        {random.name, sparse.name, random.header, sparse.header, openturbo::estimate_scan_dot(random.header, sparse.header)},
        {sparse.name, sparse.name, sparse.header, sparse.header, openturbo::estimate_scan_dot(sparse.header, sparse.header)}};

    openturbo::PackedTileHeader host_query_headers[4];
    openturbo::PackedTileHeader host_cache_headers[4];
    for (int i = 0; i < 4; ++i)
    {
        host_query_headers[i] = pair_cases[i].query_header;
        host_cache_headers[i] = pair_cases[i].cache_header;
    }

    openturbo::PackedTileHeader *device_query_headers = nullptr;
    openturbo::PackedTileHeader *device_cache_headers = nullptr;
    float *device_output = nullptr;
    float host_output[4] = {};

    check_cuda(cudaMalloc(&device_query_headers, sizeof(host_query_headers)), "cudaMalloc(device_query_headers)");
    check_cuda(cudaMalloc(&device_cache_headers, sizeof(host_cache_headers)), "cudaMalloc(device_cache_headers)");
    check_cuda(cudaMalloc(&device_output, sizeof(host_output)), "cudaMalloc(device_output)");

    check_cuda(cudaMemcpy(device_query_headers, host_query_headers, sizeof(host_query_headers), cudaMemcpyHostToDevice), "cudaMemcpy(query headers)");
    check_cuda(cudaMemcpy(device_cache_headers, host_cache_headers, sizeof(host_cache_headers), cudaMemcpyHostToDevice), "cudaMemcpy(cache headers)");
    check_cuda(cudaMemset(device_output, 0, sizeof(host_output)), "cudaMemset(device_output)");

    openturbo::scan_tile_dot_kernel<<<4, 32>>>(device_query_headers, device_cache_headers, device_output, 4);
    check_cuda(cudaGetLastError(), "scan_tile_dot_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check_cuda(cudaMemcpy(host_output, device_output, sizeof(host_output), cudaMemcpyDeviceToHost), "cudaMemcpy(output)");

    for (int i = 0; i < 4; ++i)
    {
        std::printf(
            "SCAN TEST: %s x %s gpu=%f ref=%f\n",
            pair_cases[i].query_name,
            pair_cases[i].cache_name,
            host_output[i],
            pair_cases[i].expected);
        require_close(host_output[i], pair_cases[i].expected, "scan kernel mismatch");
    }

    std::printf("\nPASS: GPU scan kernel matches CPU reference.\n");

    check_cuda(cudaFree(device_output), "cudaFree(device_output)");
    check_cuda(cudaFree(device_cache_headers), "cudaFree(device_cache_headers)");
    check_cuda(cudaFree(device_query_headers), "cudaFree(device_query_headers)");

    return 0;
}