#include "fwht.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>

namespace openturbo
{
    constexpr int kTileDims = 128;
    constexpr int kValuesPerLane = 4;
    constexpr float kInvSqrt2 = 0.7071067811865475f;

    struct alignas(32) PackedTileHeader
    {
        uint64_t quadrant_word_0;
        uint64_t quadrant_word_1;
        uint64_t qjl_sign_word;
        __half block_scale_fp16;
        __half local_alpha_fp16;
        uint32_t reserved_u32;
    };

    __device__ __forceinline__ float rope_angle(
        int pair_index,
        int token_pos,
        float rope_theta)
    {
        const float exponent = (2.0f * static_cast<float>(pair_index)) / static_cast<float>(kTileDims);
        const float inv_freq = __powf(rope_theta, -exponent);
        return static_cast<float>(token_pos) * inv_freq;
    }

    __device__ __forceinline__ void apply_rope_pair(
        float &x,
        float &y,
        int pair_index,
        int token_pos,
        float rope_theta)
    {
        const float angle = rope_angle(pair_index, token_pos, rope_theta);
        float s, c;
        __sincosf(angle, &s, &c);

        const float x_old = x;
        const float y_old = y;

        x = x_old * c - y_old * s;
        y = x_old * s + y_old * c;
    }

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

    __device__ __forceinline__ uint32_t sign_bit_from_value(float value)
    {
        return (value >= 0.0f) ? 1u : 0u;
    }

    __device__ __forceinline__ float sign_from_bit(uint32_t bit)
    {
        return bit ? 1.0f : -1.0f;
    }

    __device__ __forceinline__ uint32_t encode_quadrant_code(float x, float y)
    {
        const uint32_t x_bit = sign_bit_from_value(x);
        const uint32_t y_bit = sign_bit_from_value(y);
        return (x_bit << 1) | y_bit;
    }

    __device__ __forceinline__ void reconstruct_pair_from_code(
        uint32_t code,
        float scale,
        float &x_hat,
        float &y_hat)
    {
        const uint32_t y_bit = code & 0x1u;
        const uint32_t x_bit = (code >> 1) & 0x1u;

        const float sx = sign_from_bit(x_bit);
        const float sy = sign_from_bit(y_bit);
        const float center = scale * kInvSqrt2;

        x_hat = center * sx;
        y_hat = center * sy;
    }

    __device__ __forceinline__ float residual_statistic(
        float x,
        float y,
        float x_hat,
        float y_hat,
        uint32_t code)
    {
        const uint32_t y_bit = code & 0x1u;
        const uint32_t x_bit = (code >> 1) & 0x1u;

        const float sx = sign_from_bit(x_bit);
        const float sy = sign_from_bit(y_bit);

        const float ex = x - x_hat;
        const float ey = y - y_hat;

        return ex * sx + ey * sy;
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

    __global__ void encode_tile_fused_kernel(
        const float *__restrict__ input,
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

        float r0 = input[lane_base + 0];
        float r1 = input[lane_base + 1];
        float r2 = input[lane_base + 2];
        float r3 = input[lane_base + 3];

        apply_rope_pair(r0, r1, 2 * lane_id + 0, token_pos, rope_theta);
        apply_rope_pair(r2, r3, 2 * lane_id + 1, token_pos, rope_theta);

        fwht128_inplace(r0, r1, r2, r3, lane_id);

        float local_abs_max = fabsf(r0);
        local_abs_max = fmaxf(local_abs_max, fabsf(r1));
        local_abs_max = fmaxf(local_abs_max, fabsf(r2));
        local_abs_max = fmaxf(local_abs_max, fabsf(r3));

        float tile_abs_max = warp_reduce_max(local_abs_max);
        tile_abs_max = __shfl_sync(kWarpMask, tile_abs_max, 0);

        const float block_scale = tile_abs_max;

        const uint32_t q0 = encode_quadrant_code(r0, r1);
        const uint32_t q1 = encode_quadrant_code(r2, r3);

        float x0_hat, y0_hat;
        float x1_hat, y1_hat;

        reconstruct_pair_from_code(q0, block_scale, x0_hat, y0_hat);
        reconstruct_pair_from_code(q1, block_scale, x1_hat, y1_hat);

        const float rho0 = residual_statistic(r0, r1, x0_hat, y0_hat, q0);
        const float rho1 = residual_statistic(r2, r3, x1_hat, y1_hat, q1);

        const uint32_t z0 = sign_bit_from_value(rho0);
        const uint32_t z1 = sign_bit_from_value(rho1);

        const uint32_t local_quadrant_fragment = q0 | (q1 << 2);
        const uint32_t local_qjl_fragment = z0 | (z1 << 1);

        const float local_alpha_sum = fabsf(rho0) + fabsf(rho1);

        float alpha_sum_tile = warp_reduce_sum(local_alpha_sum);
        alpha_sum_tile = __shfl_sync(kWarpMask, alpha_sum_tile, 0);

        const float local_alpha = alpha_sum_tile * (1.0f / 64.0f);

        // Split-leader write: lanes 0..3 build and store the four 8-byte header words.
        const __half block_scale_fp16 = __float2half(block_scale);
        const __half local_alpha_fp16 = __float2half(local_alpha);

        const uint64_t scalar_word = pack_scalar_word(block_scale_fp16, local_alpha_fp16);

        uint64_t packed_word = 0ull;

        if (lane_id == 0)
        {
            packed_word = pack_quadrant_word_from_16_lanes(local_quadrant_fragment, 0);
        }
        else if (lane_id == 1)
        {
            packed_word = pack_quadrant_word_from_16_lanes(local_quadrant_fragment, 16);
        }
        else if (lane_id == 2)
        {
            packed_word = pack_qjl_word_from_32_lanes(local_qjl_fragment);
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

} // namespace openturbo