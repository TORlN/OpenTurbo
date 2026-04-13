#pragma once

#include <cuda_runtime.h>

namespace openturbo
{

    constexpr unsigned kWarpMask = 0xffffffffu;

    __device__ __forceinline__ float hadamard_lower(float self, float peer)
    {
        return self + peer;
    }

    __device__ __forceinline__ float hadamard_upper(float self, float peer)
    {
        return peer - self;
    }

    __device__ __forceinline__ float fwht_shuffle_stage(float value, int lane_id, int xor_mask)
    {
        const float peer = __shfl_xor_sync(kWarpMask, value, xor_mask);
        return ((lane_id & xor_mask) == 0) ? hadamard_lower(value, peer)
                                           : hadamard_upper(value, peer);
    }

    __device__ __forceinline__ void fwht128_inplace(
        float &r0,
        float &r1,
        float &r2,
        float &r3,
        int lane_id)
    {
        float a0 = r0;
        float a1 = r1;
        r0 = a0 + a1;
        r1 = a0 - a1;

        float a2 = r2;
        float a3 = r3;
        r2 = a2 + a3;
        r3 = a2 - a3;

        a0 = r0;
        a2 = r2;
        r0 = a0 + a2;
        r2 = a0 - a2;

        a1 = r1;
        a3 = r3;
        r1 = a1 + a3;
        r3 = a1 - a3;

        r0 = fwht_shuffle_stage(r0, lane_id, 1);
        r1 = fwht_shuffle_stage(r1, lane_id, 1);
        r2 = fwht_shuffle_stage(r2, lane_id, 1);
        r3 = fwht_shuffle_stage(r3, lane_id, 1);

        r0 = fwht_shuffle_stage(r0, lane_id, 2);
        r1 = fwht_shuffle_stage(r1, lane_id, 2);
        r2 = fwht_shuffle_stage(r2, lane_id, 2);
        r3 = fwht_shuffle_stage(r3, lane_id, 2);

        r0 = fwht_shuffle_stage(r0, lane_id, 4);
        r1 = fwht_shuffle_stage(r1, lane_id, 4);
        r2 = fwht_shuffle_stage(r2, lane_id, 4);
        r3 = fwht_shuffle_stage(r3, lane_id, 4);

        r0 = fwht_shuffle_stage(r0, lane_id, 8);
        r1 = fwht_shuffle_stage(r1, lane_id, 8);
        r2 = fwht_shuffle_stage(r2, lane_id, 8);
        r3 = fwht_shuffle_stage(r3, lane_id, 8);

        r0 = fwht_shuffle_stage(r0, lane_id, 16);
        r1 = fwht_shuffle_stage(r1, lane_id, 16);
        r2 = fwht_shuffle_stage(r2, lane_id, 16);
        r3 = fwht_shuffle_stage(r3, lane_id, 16);
    }

} // namespace openturbo