#include <cuda_runtime.h>

#ifdef __INTELLISENSE__
#include "../../kernels/encoder_layout.cuh"

namespace openturbo
{
    cudaError_t launch_encode_tile_fused(
        const float *input,
        PackedTileHeader *output_headers,
        int num_tiles,
        int token_pos,
        float rope_theta,
        cudaStream_t stream = nullptr);

    cudaError_t launch_encode_tile_fused_prerotated(
        const float *input,
        PackedTileHeader *output_headers,
        int num_tiles,
        cudaStream_t stream = nullptr);

    cudaError_t launch_scan_query_many_cache(
        const PackedTileHeader *query_header,
        const PackedTileHeader *cache_headers,
        float *output,
        int num_cache_tiles,
        cudaStream_t stream = nullptr);

    cudaError_t launch_scan_query_many_cache_multi_tile(
        const PackedTileHeader *query_headers,
        const PackedTileHeader *cache_headers,
        float *output,
        int num_query_tiles,
        int num_cache_tokens,
        cudaStream_t stream = nullptr);
}
#else
#include "../../kernels/openturbo_cuda_api.cuh"
#endif

#include <cstdint>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{
    void throw_on_cuda_error(cudaError_t status, const char *what)
    {
        if (status == cudaSuccess)
        {
            return;
        }

        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }

    std::uintptr_t normalize_stream_handle(std::uintptr_t stream_handle)
    {
        return stream_handle;
    }
}

PYBIND11_MODULE(_openturbo_cuda, module)
{
    module.doc() = "Minimal pybind11-facing shim for OpenTurbo CUDA launch wrappers.";

    module.def(
        "encode_tile_fused",
        [](std::uintptr_t input_ptr,
           std::uintptr_t output_headers_ptr,
           int num_tiles,
           int token_pos,
           float rope_theta,
           std::uintptr_t stream_handle)
        {
            const auto *input = reinterpret_cast<const float *>(input_ptr);
            auto *output_headers = reinterpret_cast<openturbo::PackedTileHeader *>(output_headers_ptr);
            auto stream = reinterpret_cast<cudaStream_t>(normalize_stream_handle(stream_handle));
            throw_on_cuda_error(
                openturbo::launch_encode_tile_fused(input, output_headers, num_tiles, token_pos, rope_theta, stream),
                "launch_encode_tile_fused");
        },
        py::arg("input_ptr"),
        py::arg("output_headers_ptr"),
        py::arg("num_tiles"),
        py::arg("token_pos"),
        py::arg("rope_theta"),
        py::arg("stream_handle") = static_cast<std::uintptr_t>(0));

    module.def(
        "encode_tile_fused_prerotated",
        [](std::uintptr_t input_ptr,
           std::uintptr_t output_headers_ptr,
           int num_tiles,
           std::uintptr_t stream_handle)
        {
            const auto *input = reinterpret_cast<const float *>(input_ptr);
            auto *output_headers = reinterpret_cast<openturbo::PackedTileHeader *>(output_headers_ptr);
            auto stream = reinterpret_cast<cudaStream_t>(normalize_stream_handle(stream_handle));
            throw_on_cuda_error(
                openturbo::launch_encode_tile_fused_prerotated(input, output_headers, num_tiles, stream),
                "launch_encode_tile_fused_prerotated");
        },
        py::arg("input_ptr"),
        py::arg("output_headers_ptr"),
        py::arg("num_tiles"),
        py::arg("stream_handle") = static_cast<std::uintptr_t>(0));

    module.def(
        "scan_query_many_cache",
        [](std::uintptr_t query_header_ptr,
           std::uintptr_t cache_headers_ptr,
           std::uintptr_t output_ptr,
           int num_cache_tiles,
           std::uintptr_t stream_handle)
        {
            const auto *query_header = reinterpret_cast<const openturbo::PackedTileHeader *>(query_header_ptr);
            const auto *cache_headers = reinterpret_cast<const openturbo::PackedTileHeader *>(cache_headers_ptr);
            auto *output = reinterpret_cast<float *>(output_ptr);
            auto stream = reinterpret_cast<cudaStream_t>(normalize_stream_handle(stream_handle));
            throw_on_cuda_error(
                openturbo::launch_scan_query_many_cache(query_header, cache_headers, output, num_cache_tiles, stream),
                "launch_scan_query_many_cache");
        },
        py::arg("query_header_ptr"),
        py::arg("cache_headers_ptr"),
        py::arg("output_ptr"),
        py::arg("num_cache_tiles"),
        py::arg("stream_handle") = static_cast<std::uintptr_t>(0));

    module.def(
        "scan_query_many_cache_multi_tile",
        [](std::uintptr_t query_headers_ptr,
           std::uintptr_t cache_headers_ptr,
           std::uintptr_t output_ptr,
           int num_query_tiles,
           int num_cache_tokens,
           std::uintptr_t stream_handle)
        {
            const auto *query_headers = reinterpret_cast<const openturbo::PackedTileHeader *>(query_headers_ptr);
            const auto *cache_headers = reinterpret_cast<const openturbo::PackedTileHeader *>(cache_headers_ptr);
            auto *output = reinterpret_cast<float *>(output_ptr);
            auto stream = reinterpret_cast<cudaStream_t>(normalize_stream_handle(stream_handle));
            throw_on_cuda_error(
                openturbo::launch_scan_query_many_cache_multi_tile(
                    query_headers,
                    cache_headers,
                    output,
                    num_query_tiles,
                    num_cache_tokens,
                    stream),
                "launch_scan_query_many_cache_multi_tile");
        },
        py::arg("query_headers_ptr"),
        py::arg("cache_headers_ptr"),
        py::arg("output_ptr"),
        py::arg("num_query_tiles"),
        py::arg("num_cache_tokens"),
        py::arg("stream_handle") = static_cast<std::uintptr_t>(0));
}