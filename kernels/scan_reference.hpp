#pragma once

#include "encoder_reference.hpp"

namespace openturbo
{
    inline PackedTileHeader pack_reference_header(const ReferenceHeaderFields &fields)
    {
        PackedTileHeader header{};
        header.quadrant_word_0 = fields.quadrant_word_0;
        header.quadrant_word_1 = fields.quadrant_word_1;
        header.qjl_sign_word = fields.qjl_sign_word;
        header.block_scale_fp16 = __float2half(fields.block_scale);
        header.local_alpha_fp16 = __float2half(fields.local_alpha);
        header.reserved_u32 = 0u;
        return header;
    }

    inline float estimate_scan_dot(
        const PackedTileHeader &query_header,
        const PackedTileHeader &cache_header)
    {
        const float query_scale = __half2float(query_header.block_scale_fp16);
        const float cache_scale = __half2float(cache_header.block_scale_fp16);
        const float local_alpha = __half2float(cache_header.local_alpha_fp16);

        float main_polar_dot = 0.0f;
        int qjl_correlation = 0;

        for (int pair_index = 0; pair_index < kPairsPerTile; ++pair_index)
        {
            float qx_hat;
            float qy_hat;
            float kx_hat;
            float ky_hat;

            reconstruct_pair_from_code(
                unpack_quadrant_code(query_header, pair_index),
                query_scale,
                qx_hat,
                qy_hat);
            reconstruct_pair_from_code(
                unpack_quadrant_code(cache_header, pair_index),
                cache_scale,
                kx_hat,
                ky_hat);

            main_polar_dot += qx_hat * kx_hat + qy_hat * ky_hat;
            qjl_correlation +=
                qjl_bipolar_from_bit(unpack_qjl_bit(query_header, pair_index)) *
                qjl_bipolar_from_bit(unpack_qjl_bit(cache_header, pair_index));
        }

        return main_polar_dot + local_alpha * static_cast<float>(qjl_correlation);
    }
}