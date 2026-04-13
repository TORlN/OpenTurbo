"""Minimal CUDA runtime helpers for Python smoke tests and interop.

This module is intentionally small. It gives Python tests a direct way to
allocate device memory, copy bytes, and synchronize without bringing in a large
framework dependency.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


def _candidate_cudart_paths() -> list[Path]:
    candidates: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin" / "cudart64_12.dll")

    candidates.append(Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/cudart64_12.dll"))
    return candidates


def _load_cudart() -> ctypes.WinDLL:
    if os.name != "nt":
        raise RuntimeError("cuda_runtime.py currently supports Windows only in this workspace")

    for candidate in _candidate_cudart_paths():
        if not candidate.is_file():
            continue
        return ctypes.WinDLL(str(candidate))

    raise RuntimeError("Could not locate cudart64_12.dll. Check CUDA_PATH or the CUDA 12.6 install.")


_CUDART: ctypes.WinDLL | None = None


def _cudart() -> ctypes.WinDLL:
    global _CUDART

    if _CUDART is None:
        _CUDART = _load_cudart()
        _CUDART.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        _CUDART.cudaMalloc.restype = ctypes.c_int

        _CUDART.cudaFree.argtypes = [ctypes.c_void_p]
        _CUDART.cudaFree.restype = ctypes.c_int

        _CUDART.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        _CUDART.cudaMemcpy.restype = ctypes.c_int

        _CUDART.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        _CUDART.cudaGetDeviceCount.restype = ctypes.c_int

        _CUDART.cudaDeviceSynchronize.argtypes = []
        _CUDART.cudaDeviceSynchronize.restype = ctypes.c_int

        _CUDART.cudaGetErrorString.argtypes = [ctypes.c_int]
        _CUDART.cudaGetErrorString.restype = ctypes.c_char_p

    return _CUDART


def _check_cuda(status: int, operation: str) -> None:
    if status == 0:
        return

    error = _cudart().cudaGetErrorString(status)
    message = error.decode("utf-8", errors="replace") if error else f"CUDA error {status}"
    raise RuntimeError(f"{operation} failed: {message}")


def cuda_malloc(num_bytes: int) -> int:
    if num_bytes <= 0:
        raise ValueError("num_bytes must be positive")

    ptr = ctypes.c_void_p()
    _check_cuda(_cudart().cudaMalloc(ctypes.byref(ptr), num_bytes), "cudaMalloc")
    if ptr.value is None:
        raise RuntimeError("cudaMalloc returned a null pointer")
    return int(ptr.value)


def cuda_free(device_ptr: int) -> None:
    _check_cuda(_cudart().cudaFree(ctypes.c_void_p(int(device_ptr))), "cudaFree")


def cuda_memcpy_host_to_device(device_ptr: int, host_buffer: bytes | bytearray | memoryview) -> None:
    view = memoryview(host_buffer).cast("B")
    host_array = (ctypes.c_ubyte * len(view)).from_buffer_copy(view)
    _check_cuda(
        _cudart().cudaMemcpy(
            ctypes.c_void_p(int(device_ptr)),
            ctypes.cast(host_array, ctypes.c_void_p),
            len(view),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        ),
        "cudaMemcpy host->device",
    )


def cuda_memcpy_device_to_host(device_ptr: int, num_bytes: int) -> bytes:
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")

    host_array = (ctypes.c_ubyte * num_bytes)()
    _check_cuda(
        _cudart().cudaMemcpy(
            ctypes.cast(host_array, ctypes.c_void_p),
            ctypes.c_void_p(int(device_ptr)),
            num_bytes,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        ),
        "cudaMemcpy device->host",
    )
    return bytes(host_array)


def cuda_device_synchronize() -> None:
    _check_cuda(_cudart().cudaDeviceSynchronize(), "cudaDeviceSynchronize")


def cuda_device_count() -> int:
    count = ctypes.c_int()
    _check_cuda(_cudart().cudaGetDeviceCount(ctypes.byref(count)), "cudaGetDeviceCount")
    return int(count.value)


def is_cuda_device_available() -> bool:
    try:
        return cuda_device_count() > 0
    except RuntimeError:
        return False
