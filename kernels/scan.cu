#include "encoder_layout.cuh"
#include "openturbo_cuda_api.cuh"

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

    __device__ __forceinline__ float scan_one_tile_estimate(
        const PackedTileHeader &query,
        const PackedTileHeader &cache_header,
        int lane_id)
    {
        const float query_scale = __half2float(query.block_scale_fp16);
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

        reconstruct_pair_from_code(unpack_quadrant_code(query, pair0), query_scale, qx0, qy0);
        reconstruct_pair_from_code(unpack_quadrant_code(cache_header, pair0), cache_scale, kx0, ky0);
        reconstruct_pair_from_code(unpack_quadrant_code(query, pair1), query_scale, qx1, qy1);
        reconstruct_pair_from_code(unpack_quadrant_code(cache_header, pair1), cache_scale, kx1, ky1);

        float local_dot = qx0 * kx0 + qy0 * ky0;
        local_dot += qx1 * kx1 + qy1 * ky1;

        float main_polar_dot = warp_reduce_sum(local_dot);
        main_polar_dot = __shfl_sync(0xffffffffu, main_polar_dot, 0);

        int qjl_correlation = 0;
        if (lane_id == 0)
        {
            const unsigned long long xnor = ~(query.qjl_sign_word ^ cache_header.qjl_sign_word);
            const int matches = __popcll(xnor);
            qjl_correlation = 2 * matches - kPairsPerTile;
        }
        qjl_correlation = __shfl_sync(0xffffffffu, qjl_correlation, 0);

        return main_polar_dot + local_alpha * static_cast<float>(qjl_correlation);
    }

    __global__ void scan_query_many_cache_kernel(
        const PackedTileHeader *__restrict__ query_header,
        const PackedTileHeader *__restrict__ cache_headers,
        float *__restrict__ output,
        int num_cache_tiles)
    {
        const int lane_id = threadIdx.x & 31;
        const int cache_tile_id = blockIdx.x;

        if (cache_tile_id >= num_cache_tiles)
        {
            return;
        }

        const PackedTileHeader query = query_header[0];
        const PackedTileHeader cache_header = cache_headers[cache_tile_id];
        const float estimate = scan_one_tile_estimate(query, cache_header, lane_id);

        if (lane_id == 0)
        {
            output[cache_tile_id] = estimate;
        }
    }

    __global__ void scan_query_many_cache_multi_tile_kernel(
        const PackedTileHeader *__restrict__ query_headers,
        const PackedTileHeader *__restrict__ cache_headers,
        float *__restrict__ output,
        int num_query_tiles,
        int num_cache_tokens)
    {
        const int lane_id = threadIdx.x & 31;
        const int cache_token_id = blockIdx.x;

        if (cache_token_id >= num_cache_tokens)
        {
            return;
        }

        float total = 0.0f;
        for (int tile_index = 0; tile_index < num_query_tiles; ++tile_index)
        {
            const PackedTileHeader query = query_headers[tile_index];
            const PackedTileHeader cache = cache_headers[cache_token_id * num_query_tiles + tile_index];
            total += scan_one_tile_estimate(query, cache, lane_id);
        }

        if (lane_id == 0)
        {
            output[cache_token_id] = total;
        }
    }

    cudaError_t launch_scan_query_many_cache(
        const PackedTileHeader *query_header,
        const PackedTileHeader *cache_headers,
        float *output,
        int num_cache_tiles,
        cudaStream_t stream)
    {
        if (num_cache_tiles <= 0)
        {
            return cudaSuccess;
        }

        scan_query_many_cache_kernel<<<num_cache_tiles, 32, 0, stream>>>(
            query_header,
            cache_headers,
            output,
            num_cache_tiles);
        return cudaGetLastError();
    }

    cudaError_t launch_scan_query_many_cache_multi_tile(
        const PackedTileHeader *query_headers,
        const PackedTileHeader *cache_headers,
        float *output,
        int num_query_tiles,
        int num_cache_tokens,
        cudaStream_t stream)
    {
        if (num_query_tiles <= 0 || num_cache_tokens <= 0)
        {
            return cudaSuccess;
        }

        scan_query_many_cache_multi_tile_kernel<<<num_cache_tokens, 32, 0, stream>>>(
            query_headers,
            cache_headers,
            output,
            num_query_tiles,
            num_cache_tokens);
        return cudaGetLastError();
    }
}