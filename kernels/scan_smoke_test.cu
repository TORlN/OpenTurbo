#include "encoder_layout.cuh"
#include "openturbo_cuda_api.cuh"
#include "encoder_reference.hpp"
#include "scan_reference.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace
{
    constexpr float kScanAbsTolerance = 1e-2f;
    constexpr float kScanRelTolerance = 1e-5f;
    constexpr int kBenchmarkIterations = 200;

    struct TileCase
    {
        const char *name;
        openturbo::PackedTileHeader header;
        std::array<float, openturbo::kTileDims> input;
    };

    struct ScanPairCase
    {
        const char *query_name;
        const char *cache_name;
        openturbo::PackedTileHeader cache_header;
        float expected;
    };

    struct MultiTileScanCase
    {
        const char *name;
        int num_query_tiles;
        int num_cache_tokens;
        openturbo::PackedTileHeader query_headers[4];
        openturbo::PackedTileHeader cache_headers[16];
        float expected[4];
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
        for (int index = 0; index < openturbo::kTileDims; ++index)
        {
            tile_case.input[static_cast<size_t>(index)] = input[index];
        }
        return tile_case;
    }

    float compute_fwht_dot(const TileCase &lhs, const TileCase &rhs)
    {
        float lhs_transformed[openturbo::kTileDims];
        float rhs_transformed[openturbo::kTileDims];
        for (int index = 0; index < openturbo::kTileDims; ++index)
        {
            lhs_transformed[index] = lhs.input[static_cast<size_t>(index)];
            rhs_transformed[index] = rhs.input[static_cast<size_t>(index)];
        }

        openturbo::fwht128_cpu(lhs_transformed);
        openturbo::fwht128_cpu(rhs_transformed);

        float total = 0.0f;
        for (int index = 0; index < openturbo::kTileDims; ++index)
        {
            total += lhs_transformed[index] * rhs_transformed[index];
        }
        return total;
    }

    void print_reference_calibration(const TileCase &query_case, const TileCase &cache_case)
    {
        const float fwht_exact = compute_fwht_dot(query_case, cache_case);
        const float current_estimate = openturbo::estimate_scan_dot(query_case.header, cache_case.header);
        const float legacy_corner_estimate = openturbo::estimate_scan_dot_corner(query_case.header, cache_case.header);

        std::printf(
            "CALIBRATION: %s x %s fwht_exact=%f current=%f legacy_corner=%f current_abs_error=%f legacy_corner_abs_error=%f\n",
            query_case.name,
            cache_case.name,
            fwht_exact,
            current_estimate,
            legacy_corner_estimate,
            std::fabs(current_estimate - fwht_exact),
            std::fabs(legacy_corner_estimate - fwht_exact));
    }

    MultiTileScanCase make_multi_tile_case(
        const TileCase &query_a,
        const TileCase &query_b,
        const TileCase &cache_token0_a,
        const TileCase &cache_token0_b,
        const TileCase &cache_token1_a,
        const TileCase &cache_token1_b,
        const TileCase &cache_token2_a,
        const TileCase &cache_token2_b)
    {
        MultiTileScanCase scan_case{};
        scan_case.name = "Two-Tile Head Scan";
        scan_case.num_query_tiles = 2;
        scan_case.num_cache_tokens = 3;

        scan_case.query_headers[0] = query_a.header;
        scan_case.query_headers[1] = query_b.header;

        scan_case.cache_headers[0] = cache_token0_a.header;
        scan_case.cache_headers[1] = cache_token0_b.header;
        scan_case.cache_headers[2] = cache_token1_a.header;
        scan_case.cache_headers[3] = cache_token1_b.header;
        scan_case.cache_headers[4] = cache_token2_a.header;
        scan_case.cache_headers[5] = cache_token2_b.header;

        scan_case.expected[0] = openturbo::estimate_scan_dot_multi_tile(scan_case.query_headers, &scan_case.cache_headers[0], scan_case.num_query_tiles);
        scan_case.expected[1] = openturbo::estimate_scan_dot_multi_tile(scan_case.query_headers, &scan_case.cache_headers[2], scan_case.num_query_tiles);
        scan_case.expected[2] = openturbo::estimate_scan_dot_multi_tile(scan_case.query_headers, &scan_case.cache_headers[4], scan_case.num_query_tiles);
        return scan_case;
    }

    void run_scan_batch(const TileCase &query_case, const ScanPairCase *pair_cases, int count)
    {
        openturbo::PackedTileHeader host_cache_headers[4];
        for (int i = 0; i < count; ++i)
        {
            host_cache_headers[i] = pair_cases[i].cache_header;
        }

        openturbo::PackedTileHeader *device_query_header = nullptr;
        openturbo::PackedTileHeader *device_cache_headers = nullptr;
        float *device_output = nullptr;
        float host_output[4] = {};

        check_cuda(cudaMalloc(&device_query_header, sizeof(openturbo::PackedTileHeader)), "cudaMalloc(device_query_header)");
        check_cuda(cudaMalloc(&device_cache_headers, sizeof(host_cache_headers)), "cudaMalloc(device_cache_headers)");
        check_cuda(cudaMalloc(&device_output, sizeof(host_output)), "cudaMalloc(device_output)");

        check_cuda(cudaMemcpy(device_query_header, &query_case.header, sizeof(query_case.header), cudaMemcpyHostToDevice), "cudaMemcpy(query header)");
        check_cuda(cudaMemcpy(device_cache_headers, host_cache_headers, sizeof(host_cache_headers), cudaMemcpyHostToDevice), "cudaMemcpy(cache headers)");
        check_cuda(cudaMemset(device_output, 0, sizeof(host_output)), "cudaMemset(device_output)");

        check_cuda(
            openturbo::launch_scan_query_many_cache(device_query_header, device_cache_headers, device_output, count),
            "launch_scan_query_many_cache");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        check_cuda(cudaMemcpy(host_output, device_output, sizeof(host_output), cudaMemcpyDeviceToHost), "cudaMemcpy(output)");

        std::printf("\nQUERY BATCH: %s vs %d cache tiles\n", query_case.name, count);
        for (int i = 0; i < count; ++i)
        {
            std::printf(
                "SCAN TEST: %s x %s gpu=%f ref=%f\n",
                pair_cases[i].query_name,
                pair_cases[i].cache_name,
                host_output[i],
                pair_cases[i].expected);
            require_close(host_output[i], pair_cases[i].expected, "scan kernel mismatch");
        }

        check_cuda(cudaFree(device_output), "cudaFree(device_output)");
        check_cuda(cudaFree(device_cache_headers), "cudaFree(device_cache_headers)");
        check_cuda(cudaFree(device_query_header), "cudaFree(device_query_header)");
    }

    void run_multi_tile_scan_case(const MultiTileScanCase &scan_case)
    {
        openturbo::PackedTileHeader *device_query_headers = nullptr;
        openturbo::PackedTileHeader *device_cache_headers = nullptr;
        float *device_output = nullptr;
        float host_output[4] = {};

        const size_t query_bytes = static_cast<size_t>(scan_case.num_query_tiles) * sizeof(openturbo::PackedTileHeader);
        const size_t cache_bytes = static_cast<size_t>(scan_case.num_query_tiles) * static_cast<size_t>(scan_case.num_cache_tokens) * sizeof(openturbo::PackedTileHeader);
        const size_t output_bytes = static_cast<size_t>(scan_case.num_cache_tokens) * sizeof(float);

        check_cuda(cudaMalloc(&device_query_headers, query_bytes), "cudaMalloc(device_query_headers)");
        check_cuda(cudaMalloc(&device_cache_headers, cache_bytes), "cudaMalloc(device_cache_headers)");
        check_cuda(cudaMalloc(&device_output, output_bytes), "cudaMalloc(device_output)");

        check_cuda(cudaMemcpy(device_query_headers, scan_case.query_headers, query_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(multi query headers)");
        check_cuda(cudaMemcpy(device_cache_headers, scan_case.cache_headers, cache_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(multi cache headers)");
        check_cuda(cudaMemset(device_output, 0, output_bytes), "cudaMemset(multi output)");

        check_cuda(
            openturbo::launch_scan_query_many_cache_multi_tile(
                device_query_headers,
                device_cache_headers,
                device_output,
                scan_case.num_query_tiles,
                scan_case.num_cache_tokens),
            "launch_scan_query_many_cache_multi_tile");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        check_cuda(cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(multi output)");

        std::printf("\nMULTI-TILE CASE: %s\n", scan_case.name);
        for (int i = 0; i < scan_case.num_cache_tokens; ++i)
        {
            std::printf("MULTI SCAN TEST token %d gpu=%f ref=%f\n", i, host_output[i], scan_case.expected[i]);
            require_close(host_output[i], scan_case.expected[i], "multi-tile scan kernel mismatch");
        }

        check_cuda(cudaFree(device_output), "cudaFree(device_output)");
        check_cuda(cudaFree(device_cache_headers), "cudaFree(device_cache_headers)");
        check_cuda(cudaFree(device_query_headers), "cudaFree(device_query_headers)");
    }

    void run_benchmark_smoke(const MultiTileScanCase &scan_case)
    {
        openturbo::PackedTileHeader *device_query_headers = nullptr;
        openturbo::PackedTileHeader *device_cache_headers = nullptr;
        float *device_output = nullptr;
        cudaEvent_t start_event = nullptr;
        cudaEvent_t stop_event = nullptr;

        const int benchmark_cache_tokens = 512;
        const int query_tiles = scan_case.num_query_tiles;
        const size_t query_bytes = static_cast<size_t>(query_tiles) * sizeof(openturbo::PackedTileHeader);
        const size_t cache_bytes = static_cast<size_t>(query_tiles) * static_cast<size_t>(benchmark_cache_tokens) * sizeof(openturbo::PackedTileHeader);
        const size_t output_bytes = static_cast<size_t>(benchmark_cache_tokens) * sizeof(float);

        openturbo::PackedTileHeader host_query_headers[4] = {};
        openturbo::PackedTileHeader host_cache_headers[2048] = {};
        for (int tile_index = 0; tile_index < query_tiles; ++tile_index)
        {
            host_query_headers[tile_index] = scan_case.query_headers[tile_index];
        }
        for (int token_index = 0; token_index < benchmark_cache_tokens; ++token_index)
        {
            const int pattern = token_index % scan_case.num_cache_tokens;
            for (int tile_index = 0; tile_index < query_tiles; ++tile_index)
            {
                host_cache_headers[token_index * query_tiles + tile_index] = scan_case.cache_headers[pattern * query_tiles + tile_index];
            }
        }

        check_cuda(cudaMalloc(&device_query_headers, query_bytes), "cudaMalloc(benchmark query headers)");
        check_cuda(cudaMalloc(&device_cache_headers, cache_bytes), "cudaMalloc(benchmark cache headers)");
        check_cuda(cudaMalloc(&device_output, output_bytes), "cudaMalloc(benchmark output)");
        check_cuda(cudaMemcpy(device_query_headers, host_query_headers, query_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(benchmark query)");
        check_cuda(cudaMemcpy(device_cache_headers, host_cache_headers, cache_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(benchmark cache)");
        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(start)");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(stop)");

        check_cuda(cudaEventRecord(start_event), "cudaEventRecord(start)");
        for (int iteration = 0; iteration < kBenchmarkIterations; ++iteration)
        {
            check_cuda(
                openturbo::launch_scan_query_many_cache_multi_tile(
                    device_query_headers,
                    device_cache_headers,
                    device_output,
                    query_tiles,
                    benchmark_cache_tokens),
                "launch_scan_query_many_cache_multi_tile benchmark");
        }
        check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(stop)");
        check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");

        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime");
        const float avg_ms = elapsed_ms / static_cast<float>(kBenchmarkIterations);
        const float scans_per_second = (1000.0f * static_cast<float>(benchmark_cache_tokens)) / avg_ms;

        std::printf(
            "\nBENCHMARK: query_tiles=%d cache_tokens=%d avg_ms=%f scans_per_sec=%f\n",
            query_tiles,
            benchmark_cache_tokens,
            avg_ms,
            scans_per_second);

        check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy(stop)");
        check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy(start)");
        check_cuda(cudaFree(device_output), "cudaFree(benchmark output)");
        check_cuda(cudaFree(device_cache_headers), "cudaFree(benchmark cache)");
        check_cuda(cudaFree(device_query_headers), "cudaFree(benchmark query)");
    }
}

int main()
{
    const TileCase alternating = make_case("Alternating Ramp Tile", fill_alternating_tile);
    const TileCase random = make_case("Seeded Random Tile", fill_seeded_random_tile);
    const TileCase sparse = make_case("Sparse Spike Tile", fill_sparse_spike_tile);

    ScanPairCase alternating_query_cases[] = {
        {alternating.name, alternating.name, alternating.header, openturbo::estimate_scan_dot(alternating.header, alternating.header)},
        {alternating.name, random.name, random.header, openturbo::estimate_scan_dot(alternating.header, random.header)},
        {alternating.name, sparse.name, sparse.header, openturbo::estimate_scan_dot(alternating.header, sparse.header)}};

    ScanPairCase sparse_query_cases[] = {
        {sparse.name, alternating.name, alternating.header, openturbo::estimate_scan_dot(sparse.header, alternating.header)},
        {sparse.name, random.name, random.header, openturbo::estimate_scan_dot(sparse.header, random.header)},
        {sparse.name, sparse.name, sparse.header, openturbo::estimate_scan_dot(sparse.header, sparse.header)}};

    const MultiTileScanCase multi_tile_case = make_multi_tile_case(
        alternating,
        sparse,
        alternating,
        random,
        random,
        sparse,
        sparse,
        alternating);

    run_scan_batch(alternating, alternating_query_cases, 3);
    run_scan_batch(sparse, sparse_query_cases, 3);
    run_multi_tile_scan_case(multi_tile_case);
    run_benchmark_smoke(multi_tile_case);

    std::printf("\nREFERENCE CALIBRATION SNAPSHOT\n");
    print_reference_calibration(alternating, alternating);
    print_reference_calibration(alternating, random);
    print_reference_calibration(random, sparse);
    print_reference_calibration(sparse, sparse);

    std::printf("\nPASS: GPU scan kernel matches CPU reference.\n");

    return 0;
}