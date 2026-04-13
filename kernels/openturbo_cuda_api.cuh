#pragma once

#include "encoder_layout.cuh"

#include <cuda_runtime.h>

namespace openturbo
{
    cudaError_t launch_encode_tile_fused(
        const float *input,
        PackedTileHeader *output_headers,
        int num_tiles,
        int token_pos,
        float rope_theta,
        cudaStream_t stream = nullptr);

    cudaError_t launch_scan_query_many_cache(
        const PackedTileHeader *query_header,
        const PackedTileHeader *cache_headers,
        float *output,
        int num_cache_tiles,
        cudaStream_t stream = nullptr);
}