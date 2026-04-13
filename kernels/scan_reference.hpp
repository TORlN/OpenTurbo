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

    struct ScanDotTerms
    {
        float main_term;
        float residual_term;
    };

    inline ScanDotTerms estimate_scan_dot_terms(
        const PackedTileHeader &query_header,
        const PackedTileHeader &cache_header,
        bool use_box_center_reconstruction)
    {
        const float query_scale = __half2float(query_header.block_scale_fp16);
        const float cache_scale = __half2float(cache_header.block_scale_fp16);
        const float local_alpha = __half2float(cache_header.local_alpha_fp16);

        ScanDotTerms terms{0.0f, 0.0f};
        int qjl_correlation = 0;

        for (int pair_index = 0; pair_index < kPairsPerTile; ++pair_index)
        {
            float qx_hat;
            float qy_hat;
            float kx_hat;
            float ky_hat;

            if (use_box_center_reconstruction)
            {
                reconstruct_pair_from_code_box_center(
                    unpack_quadrant_code(query_header, pair_index),
                    query_scale,
                    qx_hat,
                    qy_hat);
                reconstruct_pair_from_code_box_center(
                    unpack_quadrant_code(cache_header, pair_index),
                    cache_scale,
                    kx_hat,
                    ky_hat);
            }
            else
            {
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
            }

            terms.main_term += qx_hat * kx_hat + qy_hat * ky_hat;
            qjl_correlation +=
                qjl_bipolar_from_bit(unpack_qjl_bit(query_header, pair_index)) *
                qjl_bipolar_from_bit(unpack_qjl_bit(cache_header, pair_index));
        }

        terms.residual_term = local_alpha * static_cast<float>(qjl_correlation);
        return terms;
    }

    inline float estimate_scan_dot(
        const PackedTileHeader &query_header,
        const PackedTileHeader &cache_header)
    {
        const ScanDotTerms terms = estimate_scan_dot_terms(query_header, cache_header, false);
        return terms.main_term + terms.residual_term;
    }

    inline float estimate_scan_dot_box_center(
        const PackedTileHeader &query_header,
        const PackedTileHeader &cache_header)
    {
        const ScanDotTerms terms = estimate_scan_dot_terms(query_header, cache_header, true);
        return terms.main_term + terms.residual_term;
    }

    inline float estimate_scan_dot_multi_tile(
        const PackedTileHeader *query_headers,
        const PackedTileHeader *cache_headers,
        int num_query_tiles)
    {
        float total = 0.0f;
        for (int tile_index = 0; tile_index < num_query_tiles; ++tile_index)
        {
            total += estimate_scan_dot(query_headers[tile_index], cache_headers[tile_index]);
        }
        return total;
    }

    inline float estimate_scan_dot_multi_tile_box_center(
        const PackedTileHeader *query_headers,
        const PackedTileHeader *cache_headers,
        int num_query_tiles)
    {
        float total = 0.0f;
        for (int tile_index = 0; tile_index < num_query_tiles; ++tile_index)
        {
            total += estimate_scan_dot_box_center(query_headers[tile_index], cache_headers[tile_index]);
        }
        return total;
    }
}