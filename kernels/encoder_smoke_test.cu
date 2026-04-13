#include "encoder_layout.cuh"
#include "openturbo_cuda_api.cuh"
#include "encoder_reference.hpp"
#include "scan_reference.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace
{
    constexpr float kFp16Tolerance = 0.5f;
    using FillTileFn = void (*)(float *, int);

    void check_cuda(cudaError_t status, const char *what)
    {
        if (status == cudaSuccess)
        {
            return;
        }

        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
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

    struct CaseResult
    {
        const char *name;
        openturbo::PackedTileHeader gpu_header;
        openturbo::PackedTileHeader packed_reference;
    };

    void print_first_pairs(const openturbo::PackedTileHeader &gpu_header, const openturbo::ReferenceHeaderFields &reference)
    {
        std::printf("\nFirst 8 pairs (GPU vs CPU reference)\n");
        for (int pair_index = 0; pair_index < 8; ++pair_index)
        {
            const uint32_t gpu_code = unpack_quadrant_code(gpu_header, pair_index);
            const uint32_t gpu_qjl = unpack_qjl_bit(gpu_header, pair_index);
            const uint32_t ref_code = (pair_index < 32)
                                          ? static_cast<uint32_t>((reference.quadrant_word_0 >> (2 * pair_index)) & 0x3ull)
                                          : static_cast<uint32_t>((reference.quadrant_word_1 >> (2 * (pair_index - 32))) & 0x3ull);
            const uint32_t ref_qjl = static_cast<uint32_t>((reference.qjl_sign_word >> pair_index) & 0x1ull);

            std::printf(
                "pair %d: quad gpu=%u cpu=%u | qjl gpu=%u cpu=%u\n",
                pair_index,
                gpu_code,
                ref_code,
                gpu_qjl,
                ref_qjl);
        }
    }

    void require_true(bool condition, const char *message)
    {
        if (condition)
        {
            return;
        }

        std::fprintf(stderr, "ASSERTION FAILED: %s\n", message);
        std::exit(1);
    }

    void validate_header(
        const openturbo::PackedTileHeader &gpu_header,
        const openturbo::ReferenceHeaderFields &reference)
    {
        const float gpu_block_scale = __half2float(gpu_header.block_scale_fp16);
        const float gpu_local_alpha = __half2float(gpu_header.local_alpha_fp16);

        require_true(
            gpu_header.quadrant_word_0 == reference.quadrant_word_0,
            "quadrant_word_0 mismatch");
        require_true(
            gpu_header.quadrant_word_1 == reference.quadrant_word_1,
            "quadrant_word_1 mismatch");
        require_true(
            gpu_header.qjl_sign_word == reference.qjl_sign_word,
            "qjl_sign_word mismatch");
        require_true(
            gpu_header.reserved_u32 == 0u,
            "reserved_u32 must remain zero");
        require_true(
            std::fabs(gpu_block_scale - reference.block_scale) <= kFp16Tolerance,
            "block_scale_fp16 outside tolerance");
        require_true(
            std::fabs(gpu_local_alpha - reference.local_alpha) <= kFp16Tolerance,
            "local_alpha_fp16 outside tolerance");

        for (int pair_index = 0; pair_index < openturbo::kPairsPerTile; ++pair_index)
        {
            const uint32_t gpu_code = unpack_quadrant_code(gpu_header, pair_index);
            const uint32_t ref_code = (pair_index < 32)
                                          ? static_cast<uint32_t>((reference.quadrant_word_0 >> (2 * pair_index)) & 0x3ull)
                                          : static_cast<uint32_t>((reference.quadrant_word_1 >> (2 * (pair_index - 32))) & 0x3ull);
            const uint32_t gpu_qjl = unpack_qjl_bit(gpu_header, pair_index);
            const uint32_t ref_qjl = static_cast<uint32_t>((reference.qjl_sign_word >> pair_index) & 0x1ull);

            require_true(gpu_code == ref_code, "quadrant code mismatch");
            require_true(gpu_qjl == ref_qjl, "QJL bit mismatch");
        }
    }

    void validate_scan_estimate(
        const CaseResult &lhs,
        const CaseResult &rhs)
    {
        const float gpu_estimate = openturbo::estimate_scan_dot(lhs.gpu_header, rhs.gpu_header);
        const float ref_estimate = openturbo::estimate_scan_dot(lhs.packed_reference, rhs.packed_reference);
        const float delta = std::fabs(gpu_estimate - ref_estimate);

        require_true(delta <= kFp16Tolerance, "scan estimator mismatch");
        std::printf(
            "SCAN PASS: %s x %s gpu=%f ref=%f\n",
            lhs.name,
            rhs.name,
            gpu_estimate,
            ref_estimate);
    }

    CaseResult run_case(const char *case_name, FillTileFn fill_tile)
    {
        float host_input[openturbo::kTileDims];
        fill_tile(host_input, openturbo::kTileDims);

        const int token_pos = 0;
        const float rope_theta = 10000.0f;
        const openturbo::ReferenceHeaderFields reference = openturbo::compute_reference_header(host_input, token_pos, rope_theta);

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

        check_cuda(
            openturbo::launch_encode_tile_fused(
                device_input,
                device_header,
                1,
                token_pos,
                rope_theta),
            "launch_encode_tile_fused");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        check_cuda(
            cudaMemcpy(&host_header, device_header, sizeof(host_header), cudaMemcpyDeviceToHost),
            "cudaMemcpy(device_header -> host_header)");

        std::printf("\n=== %s ===\n", case_name);
        std::printf("quadrant_word_0: 0x%016llx\n", static_cast<unsigned long long>(host_header.quadrant_word_0));
        std::printf("quadrant_word_1: 0x%016llx\n", static_cast<unsigned long long>(host_header.quadrant_word_1));
        std::printf("qjl_sign_word:  0x%016llx\n", static_cast<unsigned long long>(host_header.qjl_sign_word));
        std::printf("block_scale:    %f\n", __half2float(host_header.block_scale_fp16));
        std::printf("local_alpha:    %f\n", __half2float(host_header.local_alpha_fp16));
        std::printf("reserved_u32:   0x%08x\n", host_header.reserved_u32);

        std::printf("\nCPU reference\n");
        std::printf("quadrant_word_0: 0x%016llx\n", static_cast<unsigned long long>(reference.quadrant_word_0));
        std::printf("quadrant_word_1: 0x%016llx\n", static_cast<unsigned long long>(reference.quadrant_word_1));
        std::printf("qjl_sign_word:  0x%016llx\n", static_cast<unsigned long long>(reference.qjl_sign_word));
        std::printf("block_scale:    %f\n", reference.block_scale);
        std::printf("local_alpha:    %f\n", reference.local_alpha);

        print_first_pairs(host_header, reference);
        validate_header(host_header, reference);
        std::printf("\nPASS: %s matches CPU reference within FP16 tolerance.\n", case_name);

        CaseResult result{};
        result.name = case_name;
        result.gpu_header = host_header;
        result.packed_reference = openturbo::pack_reference_header(reference);

        check_cuda(cudaFree(device_header), "cudaFree(device_header)");
        check_cuda(cudaFree(device_input), "cudaFree(device_input)");

        return result;
    }
}

int main()
{
    const CaseResult alternating = run_case("Alternating Ramp Tile", fill_alternating_tile);
    const CaseResult random = run_case("Seeded Random Tile", fill_seeded_random_tile);
    const CaseResult sparse = run_case("Sparse Spike Tile", fill_sparse_spike_tile);

    std::printf("\n=== Scan Reference Checks ===\n");
    validate_scan_estimate(alternating, alternating);
    validate_scan_estimate(alternating, random);
    validate_scan_estimate(random, sparse);
    validate_scan_estimate(sparse, sparse);

    return 0;
}