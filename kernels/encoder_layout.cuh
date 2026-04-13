#pragma once

#include <cuda_fp16.h>

#include <cmath>
#include <stdint.h>

namespace openturbo
{
    constexpr int kTileDims = 128;
    constexpr int kValuesPerLane = 4;
    constexpr int kPairsPerTile = 64;
    constexpr float kInvSqrt2 = 0.7071067811865475f;
    constexpr float kBoxCenterCoord = 0.5f;
    constexpr float kBlockScaleCalibration = 0.875f;

    struct alignas(32) PackedTileHeader
    {
        uint64_t quadrant_word_0;
        uint64_t quadrant_word_1;
        uint64_t qjl_sign_word;
        __half block_scale_fp16;
        __half local_alpha_fp16;
        uint32_t reserved_u32;
    };

    __host__ __device__ __forceinline__ uint32_t sign_bit_from_value(float value)
    {
        return (value >= 0.0f) ? 1u : 0u;
    }

    __host__ __device__ __forceinline__ float sign_from_bit(uint32_t bit)
    {
        return bit ? 1.0f : -1.0f;
    }

    __host__ __device__ __forceinline__ uint32_t encode_quadrant_code(float x, float y)
    {
        return (sign_bit_from_value(x) << 1) | sign_bit_from_value(y);
    }

    __host__ __device__ __forceinline__ float rope_angle(
        int pair_index,
        int token_pos,
        float rope_theta)
    {
        const float exponent = (2.0f * static_cast<float>(pair_index)) / static_cast<float>(kTileDims);
#ifdef __CUDA_ARCH__
        const float inv_freq = __powf(rope_theta, -exponent);
#else
        const float inv_freq = std::pow(rope_theta, -exponent);
#endif
        return static_cast<float>(token_pos) * inv_freq;
    }

    __host__ __device__ __forceinline__ void apply_rope_pair(
        float &x,
        float &y,
        int pair_index,
        int token_pos,
        float rope_theta)
    {
        const float angle = rope_angle(pair_index, token_pos, rope_theta);
        float s;
        float c;
#ifdef __CUDA_ARCH__
        __sincosf(angle, &s, &c);
#else
        s = std::sin(angle);
        c = std::cos(angle);
#endif
        const float x_old = x;
        const float y_old = y;
        x = x_old * c - y_old * s;
        y = x_old * s + y_old * c;
    }

    __host__ __device__ __forceinline__ float residual_statistic_from_code(
        float x,
        float y,
        float scale,
        uint32_t code)
    {
        const uint32_t y_bit = code & 0x1u;
        const uint32_t x_bit = (code >> 1) & 0x1u;
        const float sx = sign_from_bit(x_bit);
        const float sy = sign_from_bit(y_bit);
        const float center = scale * kInvSqrt2;
        const float x_hat = center * sx;
        const float y_hat = center * sy;
        const float ex = x - x_hat;
        const float ey = y - y_hat;
        return ex * sx + ey * sy;
    }

    __host__ __device__ __forceinline__ float residual_statistic_from_code_box_center(
        float x,
        float y,
        float scale,
        uint32_t code)
    {
        const uint32_t y_bit = code & 0x1u;
        const uint32_t x_bit = (code >> 1) & 0x1u;
        const float sx = sign_from_bit(x_bit);
        const float sy = sign_from_bit(y_bit);
        const float center = scale * kBoxCenterCoord;
        const float x_hat = center * sx;
        const float y_hat = center * sy;
        const float ex = x - x_hat;
        const float ey = y - y_hat;
        return ex * sx + ey * sy;
    }

    __host__ __device__ __forceinline__ void reconstruct_pair_from_code(
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

    __host__ __device__ __forceinline__ void reconstruct_pair_from_code_box_center(
        uint32_t code,
        float scale,
        float &x_hat,
        float &y_hat)
    {
        const uint32_t y_bit = code & 0x1u;
        const uint32_t x_bit = (code >> 1) & 0x1u;
        const float sx = sign_from_bit(x_bit);
        const float sy = sign_from_bit(y_bit);
        const float center = scale * kBoxCenterCoord;
        x_hat = center * sx;
        y_hat = center * sy;
    }

    __host__ __device__ __forceinline__ int qjl_bipolar_from_bit(uint32_t bit)
    {
        return bit ? 1 : -1;
    }

    __host__ __device__ __forceinline__ uint32_t unpack_quadrant_code(
        const PackedTileHeader &header,
        int pair_index)
    {
        if (pair_index < 32)
        {
            return static_cast<uint32_t>((header.quadrant_word_0 >> (2 * pair_index)) & 0x3ull);
        }

        const int local_index = pair_index - 32;
        return static_cast<uint32_t>((header.quadrant_word_1 >> (2 * local_index)) & 0x3ull);
    }

    __host__ __device__ __forceinline__ uint32_t unpack_qjl_bit(
        const PackedTileHeader &header,
        int pair_index)
    {
        return static_cast<uint32_t>((header.qjl_sign_word >> pair_index) & 0x1ull);
    }
}