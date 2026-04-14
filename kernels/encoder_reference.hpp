#pragma once

#include "encoder_layout.cuh"

#include <cmath>

namespace openturbo
{
    struct ReferenceHeaderFields
    {
        uint64_t quadrant_word_0;
        uint64_t quadrant_word_1;
        uint64_t qjl_sign_word;
        float block_scale;
        float local_alpha;
    };

    inline void fwht128_cpu(float *values)
    {
        for (int span = 1; span < kTileDims; span <<= 1)
        {
            const int step = span << 1;
            for (int base = 0; base < kTileDims; base += step)
            {
                for (int i = 0; i < span; ++i)
                {
                    const float a = values[base + i];
                    const float b = values[base + i + span];
                    values[base + i] = a + b;
                    values[base + i + span] = a - b;
                }
            }
        }
    }

    inline ReferenceHeaderFields compute_reference_header(
        const float *input,
        int token_pos,
        float rope_theta)
    {
        float transformed[kTileDims];
        for (int i = 0; i < kTileDims; ++i)
        {
            transformed[i] = input[i];
        }

        for (int pair_index = 0; pair_index < kPairsPerTile; ++pair_index)
        {
            apply_rope_pair(
                transformed[2 * pair_index],
                transformed[2 * pair_index + 1],
                pair_index,
                token_pos,
                rope_theta);
        }

        fwht128_cpu(transformed);

        ReferenceHeaderFields fields{};
        for (int i = 0; i < kTileDims; ++i)
        {
            fields.block_scale = std::fmax(fields.block_scale, std::fabs(transformed[i]));
        }
        float tile_sq_sum = 0.0f;
        for (int i = 0; i < kTileDims; ++i)
        {
            tile_sq_sum += transformed[i] * transformed[i];
        }

        const float calibrated_abs_max = fields.block_scale * kBlockScaleCalibration;
        const float tile_rms = std::sqrt(tile_sq_sum / static_cast<float>(kTileDims));
        fields.block_scale = damp_block_scale(calibrated_abs_max, tile_rms);

        float alpha_sum = 0.0f;
        for (int pair_index = 0; pair_index < kPairsPerTile; ++pair_index)
        {
            const float x = transformed[2 * pair_index];
            const float y = transformed[2 * pair_index + 1];
            const uint32_t code = encode_quadrant_code(x, y);
            const float rho = residual_statistic_from_code_box_center(x, y, fields.block_scale, code);
            const uint32_t qjl_bit = sign_bit_from_value(rho);

            if (pair_index < 32)
            {
                fields.quadrant_word_0 |= static_cast<uint64_t>(code) << (2 * pair_index);
            }
            else
            {
                fields.quadrant_word_1 |= static_cast<uint64_t>(code) << (2 * (pair_index - 32));
            }

            fields.qjl_sign_word |= static_cast<uint64_t>(qjl_bit) << pair_index;
            alpha_sum += std::fabs(rho);
        }

        fields.local_alpha = alpha_sum * (1.0f / 64.0f);
        return fields;
    }
}