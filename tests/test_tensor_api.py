import openturbo


class FakeCudaTensor:
    def __init__(self, ptr: int, numel: int, nbytes: int, *, is_cuda: bool = True, contiguous: bool = True):
        self._ptr = ptr
        self._numel = numel
        self.nbytes = nbytes
        self.is_cuda = is_cuda
        self._contiguous = contiguous

    def data_ptr(self) -> int:
        return self._ptr

    def numel(self) -> int:
        return self._numel

    def is_contiguous(self) -> bool:
        return self._contiguous

    def element_size(self) -> int:
        return 4


def test_encode_tile_fused_tensor_rejects_non_multiple_input():
    input_tensor = FakeCudaTensor(1, 127, 127 * 4)
    output_tensor = FakeCudaTensor(2, 8, 32)

    try:
        openturbo.encode_tile_fused_tensor(input_tensor, output_tensor, token_pos=0, rope_theta=10000.0)
    except ValueError as exc:
        assert "multiple of 128" in str(exc)
    else:
        raise AssertionError("expected encode_tile_fused_tensor to reject non-128 input")


def test_scan_query_many_cache_tensor_rejects_non_cuda_tensor():
    query = FakeCudaTensor(1, 8, 32, is_cuda=False)
    cache = FakeCudaTensor(2, 16, 64)
    output = FakeCudaTensor(3, 2, 8)

    try:
        openturbo.scan_query_many_cache_tensor(query, cache, output)
    except ValueError as exc:
        assert "CUDA-backed" in str(exc)
    else:
        raise AssertionError("expected scan_query_many_cache_tensor to reject a non-CUDA tensor")
