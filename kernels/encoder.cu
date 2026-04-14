#include "fwht.cuh"
#include "encoder_layout.cuh"
#include "openturbo_cuda_api.cuh"

#include <cuda_runtime.h>

#include <math.h>

namespace openturbo
{
    constexpr int kWarpsPerBlock = 4;
    constexpr int kThreadsPerBlock = 32 * kWarpsPerBlock;

    struct LaneValues
    {
        float r0;
        float r1;
        float r2;
        float r3;
    };

    struct LaneQuantization
    {
        uint32_t quadrant_fragment;
        uint32_t qjl_fragment;
        float alpha_sum;
    };

    __device__ __forceinline__ float warp_reduce_max(float value)
    {
        value = fmaxf(value, __shfl_down_sync(kWarpMask, value, 16));
        value = fmaxf(value, __shfl_down_sync(kWarpMask, value, 8));
        value = fmaxf(value, __shfl_down_sync(kWarpMask, value, 4));
        value = fmaxf(value, __shfl_down_sync(kWarpMask, value, 2));
        value = fmaxf(value, __shfl_down_sync(kWarpMask, value, 1));
        return value;
    }

    __device__ __forceinline__ float warp_reduce_sum(float value)
    {
        value += __shfl_down_sync(kWarpMask, value, 16);
        value += __shfl_down_sync(kWarpMask, value, 8);
        value += __shfl_down_sync(kWarpMask, value, 4);
        value += __shfl_down_sync(kWarpMask, value, 2);
        value += __shfl_down_sync(kWarpMask, value, 1);
        return value;
    }

    __device__ __forceinline__ LaneValues load_lane_values(
        const float *__restrict__ input,
        int lane_base)
    {
        LaneValues values{};
        values.r0 = input[lane_base + 0];
        values.r1 = input[lane_base + 1];
        values.r2 = input[lane_base + 2];
        values.r3 = input[lane_base + 3];
        return values;
    }

    __device__ __forceinline__ LaneValues load_lane_values(
        const __half *__restrict__ input,
        int lane_base)
    {
        LaneValues values{};
        values.r0 = __half2float(input[lane_base + 0]);
        values.r1 = __half2float(input[lane_base + 1]);
        values.r2 = __half2float(input[lane_base + 2]);
        values.r3 = __half2float(input[lane_base + 3]);
        return values;
    }

    __device__ __forceinline__ void apply_rope_to_lane(
        LaneValues &values,
        int lane_id,
        int token_pos,
        float rope_theta)
    {
        apply_rope_pair(values.r0, values.r1, 2 * lane_id + 0, token_pos, rope_theta);
        apply_rope_pair(values.r2, values.r3, 2 * lane_id + 1, token_pos, rope_theta);
    }

    __device__ __forceinline__ void apply_fwht_to_lane(LaneValues &values, int lane_id)
    {
        fwht128_inplace(values.r0, values.r1, values.r2, values.r3, lane_id);
    }

    template <bool ApplyRope>
    __device__ __forceinline__ void maybe_apply_rope_to_lane(
        LaneValues &values,
        int lane_id,
        int token_pos,
        float rope_theta)
    {
        if constexpr (ApplyRope)
        {
            apply_rope_to_lane(values, lane_id, token_pos, rope_theta);
        }
    }

    __device__ __forceinline__ float lane_abs_max(const LaneValues &values)
    {
        float local_abs_max = fabsf(values.r0);
        local_abs_max = fmaxf(local_abs_max, fabsf(values.r1));
        local_abs_max = fmaxf(local_abs_max, fabsf(values.r2));
        local_abs_max = fmaxf(local_abs_max, fabsf(values.r3));
        return local_abs_max;
    }

    __device__ __forceinline__ float lane_sq_sum(const LaneValues &values)
    {
        return values.r0 * values.r0 +
               values.r1 * values.r1 +
               values.r2 * values.r2 +
               values.r3 * values.r3;
    }

    __device__ __forceinline__ LaneQuantization quantize_lane_pairs(
        const LaneValues &values,
        float block_scale)
    {
        const uint32_t q0 = encode_quadrant_code(values.r0, values.r1);
        const uint32_t q1 = encode_quadrant_code(values.r2, values.r3);

        const float rho0 = residual_statistic_from_code_box_center(values.r0, values.r1, block_scale, q0);
        const float rho1 = residual_statistic_from_code_box_center(values.r2, values.r3, block_scale, q1);

        LaneQuantization result{};
        result.quadrant_fragment = q0 | (q1 << 2);
        result.qjl_fragment = sign_bit_from_value(rho0) | (sign_bit_from_value(rho1) << 1);
        result.alpha_sum = fabsf(rho0) + fabsf(rho1);
        return result;
    }

    static_assert(sizeof(PackedTileHeader) == 32, "PackedTileHeader must be exactly 32 bytes.");
    static_assert(alignof(PackedTileHeader) == 32, "PackedTileHeader must be 32-byte aligned.");

    __device__ __forceinline__ uint64_t pack_quadrant_word_from_16_lanes(
        uint32_t local_quadrant_fragment,
        int source_lane_base)
    {
        uint64_t word = 0ull;

#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            const uint32_t fragment =
                __shfl_sync(kWarpMask, local_quadrant_fragment, source_lane_base + i) & 0xFu;
            word |= static_cast<uint64_t>(fragment) << (4 * i);
        }

        return word;
    }

    __device__ __forceinline__ uint64_t pack_qjl_word_from_32_lanes(
        uint32_t local_qjl_fragment)
    {
        uint64_t word = 0ull;

#pragma unroll
        for (int i = 0; i < 32; ++i)
        {
            const uint32_t fragment =
                __shfl_sync(kWarpMask, local_qjl_fragment, i) & 0x3u;
            word |= static_cast<uint64_t>(fragment) << (2 * i);
        }

        return word;
    }

    __device__ __forceinline__ uint64_t pack_scalar_word(
        __half block_scale,
        __half local_alpha)
    {
        const uint64_t scale_bits = static_cast<uint64_t>(__half_as_ushort(block_scale));
        const uint64_t alpha_bits = static_cast<uint64_t>(__half_as_ushort(local_alpha));

        return scale_bits | (alpha_bits << 16);
    }

    template <bool ApplyRope, typename InputType>
    __global__ void encode_tile_fused_kernel(
        const InputType *__restrict__ input,
        PackedTileHeader *__restrict__ output_headers,
        int num_tiles,
        int token_pos,
        float rope_theta)
    {
        const int lane_id = threadIdx.x & 31;
        const int warp_id_in_block = threadIdx.x >> 5;
        const int warps_per_block = blockDim.x >> 5;
        const int tile_id = blockIdx.x * warps_per_block + warp_id_in_block;

        if (tile_id >= num_tiles)
        {
            return;
        }

        const int tile_base = tile_id * kTileDims;
        const int lane_base = tile_base + lane_id * kValuesPerLane;

        LaneValues values = load_lane_values(input, lane_base);
        maybe_apply_rope_to_lane<ApplyRope>(values, lane_id, token_pos, rope_theta);
        apply_fwht_to_lane(values, lane_id);

        const float local_abs_max = lane_abs_max(values);
        const float local_sq_sum = lane_sq_sum(values);

        float tile_abs_max = warp_reduce_max(local_abs_max);
        tile_abs_max = __shfl_sync(kWarpMask, tile_abs_max, 0);

        float tile_sq_sum = warp_reduce_sum(local_sq_sum);
        tile_sq_sum = __shfl_sync(kWarpMask, tile_sq_sum, 0);

        const float tile_rms = sqrtf(tile_sq_sum * (1.0f / static_cast<float>(kTileDims)));

        const float calibrated_abs_max = tile_abs_max * kBlockScaleCalibration;
        const float block_scale = damp_block_scale(calibrated_abs_max, tile_rms);

        const LaneQuantization quantized = quantize_lane_pairs(values, block_scale);

        float alpha_sum_tile = warp_reduce_sum(quantized.alpha_sum);
        alpha_sum_tile = __shfl_sync(kWarpMask, alpha_sum_tile, 0);

        const float local_alpha = alpha_sum_tile * (1.0f / 64.0f);

        // Split-leader write: lanes 0..3 build and store the four 8-byte header words.
        const __half block_scale_fp16 = __float2half(block_scale);
        const __half local_alpha_fp16 = __float2half(local_alpha);

        const uint64_t scalar_word = pack_scalar_word(block_scale_fp16, local_alpha_fp16);
        const uint64_t quadrant_word_0 = pack_quadrant_word_from_16_lanes(quantized.quadrant_fragment, 0);
        const uint64_t quadrant_word_1 = pack_quadrant_word_from_16_lanes(quantized.quadrant_fragment, 16);
        const uint64_t qjl_sign_word = pack_qjl_word_from_32_lanes(quantized.qjl_fragment);

        uint64_t packed_word = 0ull;

        if (lane_id == 0)
        {
            packed_word = quadrant_word_0;
        }
        else if (lane_id == 1)
        {
            packed_word = quadrant_word_1;
        }
        else if (lane_id == 2)
        {
            packed_word = qjl_sign_word;
        }
        else if (lane_id == 3)
        {
            packed_word = scalar_word;
        }

        if (lane_id < 4)
        {
            // The header is a fixed 32-byte payload viewed as four aligned 64-bit words.
            uint64_t *header_words =
                reinterpret_cast<uint64_t *>(&output_headers[tile_id]);
            header_words[lane_id] = packed_word;
        }
    }

    cudaError_t launch_encode_tile_fused(
        const float *input,
        PackedTileHeader *output_headers,
        int num_tiles,
        int token_pos,
        float rope_theta,
        cudaStream_t stream)
    {
        if (num_tiles <= 0)
        {
            return cudaSuccess;
        }

        const int blocks = (num_tiles + kWarpsPerBlock - 1) / kWarpsPerBlock;
        encode_tile_fused_kernel<true, float><<<blocks, kThreadsPerBlock, 0, stream>>>(
            input,
            output_headers,
            num_tiles,
            token_pos,
            rope_theta);
        return cudaGetLastError();
    }

    cudaError_t launch_encode_tile_fused_prerotated(
        const float *input,
        PackedTileHeader *output_headers,
        int num_tiles,
        cudaStream_t stream)
    {
        if (num_tiles <= 0)
        {
            return cudaSuccess;
        }

        const int blocks = (num_tiles + kWarpsPerBlock - 1) / kWarpsPerBlock;
        encode_tile_fused_kernel<false, float><<<blocks, kThreadsPerBlock, 0, stream>>>(
            input,
            output_headers,
            num_tiles,
            0,
            0.0f);
        return cudaGetLastError();
    }

    cudaError_t launch_encode_tile_fused(
        const __half *input,
        PackedTileHeader *output_headers,
        int num_tiles,
        int token_pos,
        float rope_theta,
        cudaStream_t stream)
    {
        if (num_tiles <= 0)
        {
            return cudaSuccess;
        }

        const int blocks = (num_tiles + kWarpsPerBlock - 1) / kWarpsPerBlock;
        encode_tile_fused_kernel<true, __half><<<blocks, kThreadsPerBlock, 0, stream>>>(
            input,
            output_headers,
            num_tiles,
            token_pos,
            rope_theta);
        return cudaGetLastError();
    }

    cudaError_t launch_encode_tile_fused_prerotated(
        const __half *input,
        PackedTileHeader *output_headers,
        int num_tiles,
        cudaStream_t stream)
    {
        if (num_tiles <= 0)
        {
            return cudaSuccess;
        }

        const int blocks = (num_tiles + kWarpsPerBlock - 1) / kWarpsPerBlock;
        encode_tile_fused_kernel<false, __half><<<blocks, kThreadsPerBlock, 0, stream>>>(
            input,
            output_headers,
            num_tiles,
            0,
            0.0f);
        return cudaGetLastError();
    }

} // namespace openturbo