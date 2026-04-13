#include "encoder_layout.cuh"

#include <cuda_runtime.h>

#include <math.h>

namespace openturbo
{
    __device__ __forceinline__ float warp_reduce_sum(float value)
    {
        value += __shfl_down_sync(0xffffffffu, value, 16);
        value += __shfl_down_sync(0xffffffffu, value, 8);
        value += __shfl_down_sync(0xffffffffu, value, 4);
        value += __shfl_down_sync(0xffffffffu, value, 2);
        value += __shfl_down_sync(0xffffffffu, value, 1);
        return value;
    }

    __global__ void scan_tile_dot_kernel(
        const PackedTileHeader *__restrict__ query_headers,
        const PackedTileHeader *__restrict__ cache_headers,
        float *__restrict__ output,
        int num_pairs)
    {
        const int lane_id = threadIdx.x & 31;
        const int tile_id = blockIdx.x;

        if (tile_id >= num_pairs)
        {
            return;
        }

        const PackedTileHeader query_header = query_headers[tile_id];
        const PackedTileHeader cache_header = cache_headers[tile_id];
        const float query_scale = __half2float(query_header.block_scale_fp16);
        const float cache_scale = __half2float(cache_header.block_scale_fp16);
        const float local_alpha = __half2float(cache_header.local_alpha_fp16);

        const int pair0 = 2 * lane_id;
        const int pair1 = pair0 + 1;

        float qx0;
        float qy0;
        float kx0;
        float ky0;
        float qx1;
        float qy1;
        float kx1;
        float ky1;

        reconstruct_pair_from_code(unpack_quadrant_code(query_header, pair0), query_scale, qx0, qy0);
        reconstruct_pair_from_code(unpack_quadrant_code(cache_header, pair0), cache_scale, kx0, ky0);
        reconstruct_pair_from_code(unpack_quadrant_code(query_header, pair1), query_scale, qx1, qy1);
        reconstruct_pair_from_code(unpack_quadrant_code(cache_header, pair1), cache_scale, kx1, ky1);

        float local_dot = qx0 * kx0 + qy0 * ky0;
        local_dot += qx1 * kx1 + qy1 * ky1;

        float main_polar_dot = warp_reduce_sum(local_dot);
        main_polar_dot = __shfl_sync(0xffffffffu, main_polar_dot, 0);

        int qjl_correlation = 0;
        if (lane_id == 0)
        {
            const unsigned long long xnor = ~(query_header.qjl_sign_word ^ cache_header.qjl_sign_word);
            const int matches = __popcll(xnor);
            qjl_correlation = 2 * matches - kPairsPerTile;
        }
        qjl_correlation = __shfl_sync(0xffffffffu, qjl_correlation, 0);

        if (lane_id == 0)
        {
            output[tile_id] = main_polar_dot + local_alpha * static_cast<float>(qjl_correlation);
        }
    }
}