import math
import struct

import openturbo
import pytest


def test_python_cuda_encode_smoke():
    if not openturbo.is_cuda_extension_available():
        pytest.skip("CUDA extension is not available in this environment")

    values = [((index % 17) - 8) * 0.125 for index in range(128)]
    host_input = struct.pack("<128f", *values)

    input_ptr = openturbo.cuda_malloc(len(host_input))
    output_ptr = openturbo.cuda_malloc(32)
    try:
        openturbo.cuda_memcpy_host_to_device(input_ptr, host_input)
        openturbo.encode_tile_fused(
            input_ptr=input_ptr,
            output_headers_ptr=output_ptr,
            num_tiles=1,
            token_pos=19,
            rope_theta=10000.0,
        )
        openturbo.cuda_device_synchronize()
        header = openturbo.cuda_memcpy_device_to_host(output_ptr, 32)
    finally:
        openturbo.cuda_free(output_ptr)
        openturbo.cuda_free(input_ptr)

    quadrant_word_0, quadrant_word_1, _qjl_word = struct.unpack("<QQQ", header[:24])
    block_scale = struct.unpack("<e", header[24:26])[0]
    local_alpha = struct.unpack("<e", header[26:28])[0]
    reserved = struct.unpack("<I", header[28:32])[0]

    assert reserved == 0
    assert block_scale > 0.0
    assert header != bytes(32)
    assert quadrant_word_0 != 0 or quadrant_word_1 != 0
    assert math.isfinite(local_alpha)