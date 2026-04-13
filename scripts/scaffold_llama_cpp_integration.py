from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_LLAMA_DIR_NAME = "llama"


BRIDGE_HEADER = """#pragma once

#include \"ggml.h\"
#include \"openturbo/ggml_downstream.hpp\"

struct openturbo_llama_cpp_encode_request {
    const ggml_tensor * input_heads;
    ggml_tensor * output_headers_by_head;
    int token_pos;
    float rope_theta;
    bool input_is_prerotated;
    bool all_heads;
    int head_index;
    openturbo_stream_context_t stream_context;
};

struct openturbo_llama_cpp_scan_request {
    const ggml_tensor * query_headers_by_head;
    const ggml_tensor * cache_headers_by_head;
    ggml_tensor * output_by_head;
    int num_query_tiles;
    int num_cache_tokens;
    bool all_heads;
    int head_index;
    openturbo_stream_context_t stream_context;
};

openturbo_status_t openturbo_llama_cpp_encode(
    const openturbo_llama_cpp_encode_request * request,
    int * cuda_status_out);

openturbo_status_t openturbo_llama_cpp_scan(
    const openturbo_llama_cpp_scan_request * request,
    int * cuda_status_out);

void openturbo_llama_cpp_probe_k_cache_write(
    const ggml_tensor * k_cur_3d,
    const ggml_tensor * k_cache,
    int32_t layer_index,
    int64_t kv_size,
    int64_t n_stream);
"""


BRIDGE_CPP = """#include \"openturbo_llama_cpp_bridge.hpp\"

#include <cstdio>

namespace {
bool openturbo_is_k_probe_compatible(const ggml_tensor * k_cur_3d) {
    if (k_cur_3d == nullptr) {
        return false;
    }

    return k_cur_3d->type == GGML_TYPE_F32 && k_cur_3d->ne[0] > 0 && (k_cur_3d->ne[0] % OPENTURBO_TILE_DIMS) == 0 &&
           k_cur_3d->ne[1] > 0 && k_cur_3d->ne[2] > 0 &&
           ggml_row_size(k_cur_3d->type, k_cur_3d->ne[0]) == k_cur_3d->nb[1];
}
}  // namespace

#ifndef OPENTURBO_EXPERIMENTAL_K_CACHE_PROBE_ONLY

openturbo_status_t openturbo_llama_cpp_encode(
    const openturbo_llama_cpp_encode_request * request,
    int * cuda_status_out) {
    if (request == nullptr || request->input_heads == nullptr || request->output_headers_by_head == nullptr) {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    if (request->input_is_prerotated) {
        if (request->all_heads) {
            return openturbo::ggml_downstream::llama_encode_all_heads_from_ggml_tensors_prerotated(
                request->input_heads,
                request->output_headers_by_head,
                request->stream_context,
                cuda_status_out);
        }

        return openturbo::ggml_downstream::llama_encode_from_ggml_tensors_prerotated(
            request->input_heads,
            request->output_headers_by_head,
            request->head_index,
            request->stream_context,
            cuda_status_out);
    }

    if (request->all_heads) {
        return openturbo::ggml_downstream::llama_encode_all_heads_from_ggml_tensors(
            request->input_heads,
            request->output_headers_by_head,
            request->token_pos,
            request->rope_theta,
            request->stream_context,
            cuda_status_out);
    }

    return openturbo::ggml_downstream::llama_encode_from_ggml_tensors(
        request->input_heads,
        request->output_headers_by_head,
        request->head_index,
        request->token_pos,
        request->rope_theta,
        request->stream_context,
        cuda_status_out);
}

openturbo_status_t openturbo_llama_cpp_scan(
    const openturbo_llama_cpp_scan_request * request,
    int * cuda_status_out) {
    if (request == nullptr || request->query_headers_by_head == nullptr ||
        request->cache_headers_by_head == nullptr || request->output_by_head == nullptr) {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
    }

    if (request->all_heads) {
        return openturbo::ggml_downstream::llama_scan_all_heads_from_ggml_tensors(
            request->query_headers_by_head,
            request->cache_headers_by_head,
            request->output_by_head,
            request->num_query_tiles,
            request->num_cache_tokens,
            request->stream_context,
            cuda_status_out);
    }

    return openturbo::ggml_downstream::llama_scan_from_ggml_tensors(
        request->query_headers_by_head,
        request->cache_headers_by_head,
        request->output_by_head,
        request->head_index,
        request->num_query_tiles,
        request->num_cache_tokens,
        request->stream_context,
        cuda_status_out);
}

#endif

void openturbo_llama_cpp_probe_k_cache_write(
    const ggml_tensor * k_cur_3d,
    const ggml_tensor * k_cache,
    int32_t layer_index,
    int64_t kv_size,
    int64_t n_stream) {
    static bool logged_once = false;
    if (logged_once || k_cur_3d == nullptr || k_cache == nullptr) {
        return;
    }

    const bool compatible = openturbo_is_k_probe_compatible(k_cur_3d);

    std::fprintf(
        stderr,
        "[openturbo] cpy_k probe layer=%d compatible=%d head_dim=%lld num_heads=%lld num_tokens=%lld "
        "num_head_tiles=%lld kv_size=%lld n_stream=%lld cache_type=%d cache_ne=[%lld,%lld,%lld]\\n",
        layer_index,
        compatible ? 1 : 0,
        static_cast<long long>(k_cur_3d->ne[0]),
        static_cast<long long>(k_cur_3d->ne[1]),
        static_cast<long long>(k_cur_3d->ne[2]),
        static_cast<long long>(k_cur_3d->ne[0] / OPENTURBO_TILE_DIMS),
        static_cast<long long>(kv_size),
        static_cast<long long>(n_stream),
        static_cast<int>(k_cache->type),
        static_cast<long long>(k_cache->ne[0]),
        static_cast<long long>(k_cache->ne[1]),
        static_cast<long long>(k_cache->ne[2]));

    logged_once = true;
}
"""


BRIDGE_README = """# OpenTurbo llama.cpp Scaffold

This scaffold is generated from OpenTurbo and is intended to live inside a llama.cpp checkout.

What it gives you:

- `openturbo_llama_cpp_bridge.hpp`: a small downstream-facing request API.
- `openturbo_llama_cpp_bridge.cpp`: wrappers that forward real `ggml_tensor` objects into OpenTurbo.
- `CMakeLists.fragment.txt`: a minimal build fragment showing where to add the bridge source.
- An optional `OPENTURBO_EXPERIMENTAL_K_CACHE_PROBE` hook that can patch `llama_kv_cache::cpy_k()` and log live tensor metadata before replacing the cache format.

What you still need to do in llama.cpp:

1. Identify the K/V cache tensors and head-local tile counts at the actual call site.
2. Fill the request structs with those tensors and the right `num_query_tiles` / `num_cache_tokens` values.
3. Link against the installed OpenTurbo headers and `openturbo_c_api` library.

Probe automation:

1. Re-run the scaffold with `--probe-k-cache`.
2. Configure llama.cpp with `-DOPENTURBO_EXPERIMENTAL_K_CACHE_PROBE=ON`.
3. If OpenTurbo is not adjacent to the checkout, also pass `-DOPENTURBO_ROOT=<path-to-openturbo>`.
4. Build and run one narrow inference path to capture the one-time `cpy_k` probe log line.

The probe patching logic targets the current llama.cpp `llama_kv_cache::cpy_k()` layout and fails fast if the expected insertion points are not present.
"""


CMAKE_FRAGMENT = """# Add this source inside the downstream llama.cpp build after OpenTurbo is available.
# Example:
# target_sources(llama PRIVATE path/to/openturbo_llama_cpp_bridge.cpp)
# target_include_directories(llama PRIVATE <OpenTurbo install prefix>/include)
# target_link_libraries(llama PRIVATE openturbo_c_api)
# Optional probe-only path:
# target_compile_definitions(llama PRIVATE OPENTURBO_EXPERIMENTAL_K_CACHE_PROBE OPENTURBO_EXPERIMENTAL_K_CACHE_PROBE_ONLY)
"""


SHADOW_CALLBACK_HEADER = """#pragma once

#include \"ggml-backend.h\"

struct openturbo_shadow_eval_callback_state {
    bool logged_write_once = false;
    bool logged_read_once = false;
    bool logged_error = false;
};

bool openturbo_shadow_cb_eval(struct ggml_tensor * tensor, bool ask, void * user_data);
"""


SHADOW_CALLBACK_CPP = """#include \"openturbo_shadow_eval_callback.hpp\"

#include \"ggml.h\"
#include \"openturbo/c_api.h\"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {
struct openturbo_shadow_layer_sidecar {
    int64_t row_capacity = 0;
    int64_t tiles_per_row = 0;
    std::vector<openturbo_packed_tile_header_t> headers;
    std::vector<uint8_t> present_rows;
};

std::unordered_map<int, openturbo_shadow_layer_sidecar> g_openturbo_shadow_sidecars;

bool openturbo_is_shadow_candidate(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->op != GGML_OP_SET_ROWS) {
        return false;
    }

    const ggml_tensor * src_values = tensor->src[0];
    const ggml_tensor * dst_cache = tensor->src[2];
    if (src_values == nullptr || dst_cache == nullptr) {
        return false;
    }

    const char * cache_name = ggml_get_name(dst_cache);
    if (cache_name == nullptr || std::strncmp(cache_name, \"cache_k_l\", 8) != 0) {
        return false;
    }

    return src_values->type == GGML_TYPE_F32 &&
           src_values->ne[0] > 0 &&
           (src_values->ne[0] % OPENTURBO_TILE_DIMS) == 0 &&
           ggml_row_size(src_values->type, src_values->ne[0]) == src_values->nb[1];
}

bool openturbo_copy_tensor_bytes(const ggml_tensor * tensor, std::vector<unsigned char> & bytes) {
    const size_t nbytes = ggml_nbytes(tensor);
    bytes.resize(nbytes);

    if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer)) {
        std::memcpy(bytes.data(), tensor->data, nbytes);
        return true;
    }

    ggml_backend_tensor_get(tensor, bytes.data(), 0, nbytes);
    return true;
}

bool openturbo_copy_row_indices(const ggml_tensor * tensor, std::vector<int64_t> & row_indices) {
    if (tensor == nullptr) {
        return false;
    }

    const int64_t count = ggml_nelements(tensor);
    if (count <= 0) {
        row_indices.clear();
        return true;
    }

    row_indices.resize(static_cast<size_t>(count));
    if (tensor->type == GGML_TYPE_I32) {
        std::vector<int32_t> raw(static_cast<size_t>(count));
        if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer)) {
            std::memcpy(raw.data(), tensor->data, sizeof(int32_t) * raw.size());
        } else {
            ggml_backend_tensor_get(tensor, raw.data(), 0, sizeof(int32_t) * raw.size());
        }

        for (size_t index = 0; index < raw.size(); ++index) {
            row_indices[index] = raw[index];
        }
        return true;
    }

    if (tensor->type == GGML_TYPE_I64) {
        if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer)) {
            std::memcpy(row_indices.data(), tensor->data, sizeof(int64_t) * row_indices.size());
        } else {
            ggml_backend_tensor_get(tensor, row_indices.data(), 0, sizeof(int64_t) * row_indices.size());
        }
        return true;
    }

    return false;
}

long long openturbo_first_row_index(const ggml_tensor * tensor) {
    if (tensor == nullptr || ggml_nelements(tensor) <= 0) {
        return -1;
    }

    if (tensor->type == GGML_TYPE_I32) {
        int32_t value = 0;
        if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer)) {
            std::memcpy(&value, tensor->data, sizeof(value));
        } else {
            ggml_backend_tensor_get(tensor, &value, 0, sizeof(value));
        }
        return value;
    }

    if (tensor->type == GGML_TYPE_I64) {
        int64_t value = 0;
        if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer)) {
            std::memcpy(&value, tensor->data, sizeof(value));
        } else {
            ggml_backend_tensor_get(tensor, &value, 0, sizeof(value));
        }
        return static_cast<long long>(value);
    }

    return -1;
}

int openturbo_parse_layer_index(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return -1;
    }

    const char * cache_name = ggml_get_name(tensor);
    if (cache_name == nullptr) {
        return -1;
    }

    int layer_index = -1;
    std::sscanf(cache_name, \"cache_k_l%d\", &layer_index);
    return layer_index;
}

int64_t openturbo_row_capacity(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return 0;
    }

    return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

openturbo_shadow_layer_sidecar & openturbo_get_or_reset_sidecar(int layer_index, int64_t row_capacity, int64_t tiles_per_row) {
    auto & sidecar = g_openturbo_shadow_sidecars[layer_index];
    if (sidecar.row_capacity != row_capacity || sidecar.tiles_per_row != tiles_per_row) {
        sidecar.row_capacity = row_capacity;
        sidecar.tiles_per_row = tiles_per_row;
        sidecar.headers.assign(static_cast<size_t>(row_capacity * tiles_per_row), {});
        sidecar.present_rows.assign(static_cast<size_t>(row_capacity), 0);
    }

    return sidecar;
}

int openturbo_store_sidecar_rows(openturbo_shadow_layer_sidecar & sidecar,
                                 const std::vector<int64_t> & row_indices,
                                 const std::vector<openturbo_packed_tile_header_t> & encoded_headers) {
    if (sidecar.tiles_per_row <= 0) {
        return 0;
    }

    int stored_rows = 0;
    const size_t tiles_per_row = static_cast<size_t>(sidecar.tiles_per_row);
    for (size_t row = 0; row < row_indices.size(); ++row) {
        const int64_t row_index = row_indices[row];
        if (row_index < 0 || row_index >= sidecar.row_capacity) {
            continue;
        }

        const size_t src_offset = row * tiles_per_row;
        const size_t dst_offset = static_cast<size_t>(row_index) * tiles_per_row;
        std::copy_n(encoded_headers.data() + src_offset, tiles_per_row, sidecar.headers.data() + dst_offset);
        sidecar.present_rows[static_cast<size_t>(row_index)] = 1;
        ++stored_rows;
    }

    return stored_rows;
}

const ggml_tensor * openturbo_find_named_tensor_prefix(const ggml_tensor * tensor,
                                                       const char *        prefix,
                                                       int                 prefix_len,
                                                       int                 depth_remaining = 8) {
    if (tensor == nullptr || depth_remaining < 0) {
        return nullptr;
    }

    const char * tensor_name = ggml_get_name(tensor);
    if (tensor_name != nullptr && std::strncmp(tensor_name, prefix, prefix_len) == 0) {
        return tensor;
    }

    for (int source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
        const ggml_tensor * source = tensor->src[source_index];
        if (source == nullptr) {
            continue;
        }

        if (const ggml_tensor * found =
                openturbo_find_named_tensor_prefix(source, prefix, prefix_len, depth_remaining - 1)) {
            return found;
        }
    }

    return nullptr;
}

const ggml_tensor * openturbo_find_named_cache_tensor(const ggml_tensor * tensor, int depth_remaining = 8) {
    return openturbo_find_named_tensor_prefix(tensor, "cache_k_l", 8, depth_remaining);
}

const ggml_tensor * openturbo_find_kq_mask_tensor(const ggml_tensor * tensor, int depth_remaining = 8) {
    return openturbo_find_named_tensor_prefix(tensor, "attn_inp_kq_mask", 16, depth_remaining);
}

bool openturbo_copy_mask_values(const ggml_tensor * tensor, std::vector<float> & mask_values) {
    if (tensor == nullptr) {
        return false;
    }

    const int64_t count = ggml_nelements(tensor);
    if (count <= 0) {
        mask_values.clear();
        return true;
    }

    mask_values.resize(static_cast<size_t>(count));
    if (tensor->type == GGML_TYPE_F32) {
        if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer)) {
            std::memcpy(mask_values.data(), tensor->data, sizeof(float) * mask_values.size());
        } else {
            ggml_backend_tensor_get(tensor, mask_values.data(), 0, sizeof(float) * mask_values.size());
        }
        return true;
    }

    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(static_cast<size_t>(count));
        if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer)) {
            std::memcpy(raw.data(), tensor->data, sizeof(ggml_fp16_t) * raw.size());
        } else {
            ggml_backend_tensor_get(tensor, raw.data(), 0, sizeof(ggml_fp16_t) * raw.size());
        }

        for (size_t index = 0; index < raw.size(); ++index) {
            mask_values[index] = ggml_fp16_to_fp32(raw[index]);
        }
        return true;
    }

    return false;
}

int64_t openturbo_rows_per_stream(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->nb[2] <= 0 || tensor->nb[3] <= 0) {
        return 0;
    }

    return tensor->nb[3] / tensor->nb[2];
}

int64_t openturbo_stream_base_index(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->nb[3] <= 0) {
        return 0;
    }

    return static_cast<int64_t>(tensor->view_offs / static_cast<size_t>(tensor->nb[3]));
}

struct openturbo_shadow_read_coverage {
    int64_t expected_rows = 0;
    int present_rows = 0;
    bool exact = false;
};

openturbo_shadow_read_coverage openturbo_compute_mask_read_coverage(
    const ggml_tensor *                    cache_tensor,
    const ggml_tensor *                    mask_tensor,
    const openturbo_shadow_layer_sidecar & sidecar) {
    openturbo_shadow_read_coverage coverage;
    if (cache_tensor == nullptr || mask_tensor == nullptr || sidecar.row_capacity <= 0) {
        return coverage;
    }

    const int64_t n_kv = mask_tensor->ne[0];
    const int64_t n_tokens_per_stream = std::max<int64_t>(mask_tensor->ne[1], 1);
    const int64_t n_stream = std::max<int64_t>(mask_tensor->ne[3], 1);
    const int64_t rows_per_stream = openturbo_rows_per_stream(cache_tensor);
    const int64_t stream_base_index = openturbo_stream_base_index(cache_tensor);
    if (n_kv <= 0 || rows_per_stream <= 0 || stream_base_index < 0) {
        return coverage;
    }

    std::vector<float> mask_values;
    if (!openturbo_copy_mask_values(mask_tensor, mask_values)) {
        return coverage;
    }

    std::vector<uint8_t> expected_row_flags(static_cast<size_t>(sidecar.row_capacity), 0);
    for (int64_t stream = 0; stream < n_stream; ++stream) {
        const int64_t global_stream_base = (stream_base_index + stream) * rows_per_stream;
        for (int64_t token = 0; token < n_tokens_per_stream; ++token) {
            const size_t token_offset = static_cast<size_t>((stream * n_tokens_per_stream + token) * n_kv);
            for (int64_t row = 0; row < n_kv; ++row) {
                const float mask_value = mask_values[token_offset + static_cast<size_t>(row)];
                if (!std::isfinite(mask_value)) {
                    continue;
                }

                const int64_t global_row = global_stream_base + row;
                if (global_row < 0 || global_row >= sidecar.row_capacity) {
                    continue;
                }

                uint8_t & expected_flag = expected_row_flags[static_cast<size_t>(global_row)];
                if (expected_flag != 0) {
                    continue;
                }

                expected_flag = 1;
                ++coverage.expected_rows;
                coverage.present_rows += sidecar.present_rows[static_cast<size_t>(global_row)] != 0 ? 1 : 0;
            }
        }
    }

    coverage.exact = true;
    return coverage;
}

bool openturbo_is_read_candidate(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }

    if (tensor->op == GGML_OP_SET_ROWS) {
        return false;
    }

    return openturbo_find_named_cache_tensor(tensor) != nullptr && openturbo_find_kq_mask_tensor(tensor) != nullptr;
}
}

bool openturbo_shadow_cb_eval(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto * state = static_cast<openturbo_shadow_eval_callback_state *>(user_data);

    if (ask) {
        const bool wants_write = openturbo_is_shadow_candidate(tensor) && (state == nullptr || !state->logged_write_once);
        const bool wants_read = openturbo_is_read_candidate(tensor) && (state == nullptr || !state->logged_read_once);
        return wants_write || wants_read;
    }

    if (openturbo_is_read_candidate(tensor)) {
        if (state != nullptr && state->logged_read_once) {
            return true;
        }

        const ggml_tensor * cache_tensor = openturbo_find_named_cache_tensor(tensor);
        const ggml_tensor * mask_tensor = openturbo_find_kq_mask_tensor(tensor);
        const int layer_index = openturbo_parse_layer_index(cache_tensor);
        const auto sidecar_it = g_openturbo_shadow_sidecars.find(layer_index);
        if (sidecar_it == g_openturbo_shadow_sidecars.end()) {
            std::fprintf(stderr,
                         \"[openturbo] shadow_read layer=%d status=missing_sidecar node=%s\\n\",
                         layer_index,
                         ggml_get_name(tensor));
            if (state != nullptr) {
                state->logged_read_once = true;
                state->logged_error = true;
            }
            return true;
        }

        const auto & sidecar = sidecar_it->second;
        const auto coverage = openturbo_compute_mask_read_coverage(cache_tensor, mask_tensor, sidecar);
        if (!coverage.exact) {
            std::fprintf(stderr,
                         \"[openturbo] shadow_read layer=%d status=mask_unavailable node=%s mask_type=%d\\n\",
                         layer_index,
                         ggml_get_name(tensor),
                         mask_tensor == nullptr ? -1 : static_cast<int>(mask_tensor->type));
            if (state != nullptr) {
                state->logged_read_once = true;
                state->logged_error = true;
            }
            return true;
        }

        std::fprintf(stderr,
                     \"[openturbo] shadow_read layer=%d status=%s node=%s expected_rows=%lld present_rows=%d tiles_per_row=%lld\\n\",
                     layer_index,
                     coverage.present_rows == coverage.expected_rows ? "success" : "partial",
                     ggml_get_name(tensor),
                     static_cast<long long>(coverage.expected_rows),
                     coverage.present_rows,
                     static_cast<long long>(sidecar.tiles_per_row));
        if (state != nullptr) {
            state->logged_read_once = true;
        }
        return true;
    }

    if (!openturbo_is_shadow_candidate(tensor)) {
        return true;
    }

    if (state != nullptr && state->logged_write_once) {
        return true;
    }

    const ggml_tensor * src_values = tensor->src[0];
    const ggml_tensor * src_indices = tensor->src[1];
    const ggml_tensor * dst_cache = tensor->src[2];
    const int layer_index = openturbo_parse_layer_index(dst_cache);

    const int64_t total_elements = ggml_nelements(src_values);
    const int num_tiles = static_cast<int>(total_elements / OPENTURBO_TILE_DIMS);
    if (num_tiles <= 0) {
        return true;
    }

    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count <= 0) {
        if (state == nullptr || !state->logged_error) {
            std::fprintf(stderr,
                         \"[openturbo] shadow_encode layer=%d status=skipped reason=no_cuda_device cuda=%d\\n\",
                         layer_index,
                         static_cast<int>(cuda_status));
            if (state != nullptr) {
                state->logged_error = true;
            }
        }
        return true;
    }

    std::vector<unsigned char> host_input_bytes;
    openturbo_copy_tensor_bytes(src_values, host_input_bytes);
    std::vector<int64_t> row_indices;
    if (!openturbo_copy_row_indices(src_indices, row_indices)) {
        std::fprintf(stderr,
                     \"[openturbo] shadow_encode layer=%d status=failed reason=unsupported_row_index_type\\n\",
                     layer_index);
        if (state != nullptr) {
            state->logged_error = true;
        }
        return true;
    }

    if (row_indices.empty()) {
        return true;
    }

    if (num_tiles % static_cast<int>(row_indices.size()) != 0) {
        std::fprintf(stderr,
                     \"[openturbo] shadow_encode layer=%d status=failed reason=non_integral_tiles_per_row num_tiles=%d rows=%zu\\n\",
                     layer_index,
                     num_tiles,
                     row_indices.size());
        if (state != nullptr) {
            state->logged_error = true;
        }
        return true;
    }

    const int64_t row_capacity = openturbo_row_capacity(dst_cache);
    const int64_t tiles_per_row = num_tiles / static_cast<int64_t>(row_indices.size());

    float * device_input = nullptr;
    openturbo_packed_tile_header_t * device_headers = nullptr;
    openturbo_packed_tile_header_t first_header{};

    cuda_status = cudaMalloc(&device_input, host_input_bytes.size());
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr,
                     \"[openturbo] shadow_encode layer=%d status=cuda_alloc_failed cuda=%d\\n\",
                     openturbo_parse_layer_index(dst_cache),
                     static_cast<int>(cuda_status));
        if (state != nullptr) {
            state->logged_error = true;
        }
        return true;
    }

    cuda_status = cudaMalloc(&device_headers, sizeof(openturbo_packed_tile_header_t) * static_cast<size_t>(num_tiles));
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaMemcpy(device_input, host_input_bytes.data(), host_input_bytes.size(), cudaMemcpyHostToDevice);
    }

    int openturbo_cuda_status = 0;
    openturbo_status_t status = OPENTURBO_STATUS_CUDA_ERROR;
    if (cuda_status == cudaSuccess) {
        status = openturbo_encode_tile_fused_prerotated(
            device_input,
            device_headers,
            num_tiles,
            nullptr,
            &openturbo_cuda_status);
    }

    if (status == OPENTURBO_STATUS_SUCCESS) {
        cuda_status = cudaMemcpy(&first_header, device_headers, sizeof(first_header), cudaMemcpyDeviceToHost);
    }

    std::vector<openturbo_packed_tile_header_t> host_headers(static_cast<size_t>(num_tiles));
    if (status == OPENTURBO_STATUS_SUCCESS && cuda_status == cudaSuccess) {
        cuda_status = cudaMemcpy(
            host_headers.data(),
            device_headers,
            sizeof(openturbo_packed_tile_header_t) * host_headers.size(),
            cudaMemcpyDeviceToHost);
    }

    if (device_headers != nullptr) {
        cudaFree(device_headers);
    }
    if (device_input != nullptr) {
        cudaFree(device_input);
    }

    if (status == OPENTURBO_STATUS_SUCCESS && cuda_status == cudaSuccess) {
        auto & sidecar = openturbo_get_or_reset_sidecar(layer_index, row_capacity, tiles_per_row);
        const int stored_rows = openturbo_store_sidecar_rows(sidecar, row_indices, host_headers);
        std::fprintf(stderr,
                     \"[openturbo] shadow_encode layer=%d status=success num_tiles=%d tiles_per_row=%lld first_row=%lld stored_rows=%d sidecar_rows=%lld cache=%s first_qword0=%llu\\n\",
                     layer_index,
                     num_tiles,
                     static_cast<long long>(tiles_per_row),
                     openturbo_first_row_index(src_indices),
                     stored_rows,
                     static_cast<long long>(sidecar.row_capacity),
                     ggml_get_name(dst_cache),
                     static_cast<unsigned long long>(first_header.quadrant_word_0));
        if (state != nullptr) {
            state->logged_write_once = true;
        }
        return true;
    }

    std::fprintf(stderr,
                 \"[openturbo] shadow_encode layer=%d status=failed openturbo=%d cuda=%d detail=%s\\n\",
                 layer_index,
                 static_cast<int>(status),
                 status == OPENTURBO_STATUS_SUCCESS ? static_cast<int>(cuda_status) : openturbo_cuda_status,
                 status == OPENTURBO_STATUS_SUCCESS ? cudaGetErrorString(cuda_status) : openturbo_status_string(status));
    if (state != nullptr) {
        state->logged_error = true;
    }
    return true;
}
"""


PROBE_OPTION_BLOCK = """option(OPENTURBO_EXPERIMENTAL_K_CACHE_PROBE \"llama: enable OpenTurbo K-cache probe logging\" OFF)
set(OPENTURBO_ROOT \"\" CACHE PATH \"Path to the OpenTurbo workspace or install prefix\")

"""


PROBE_HELPER_BLOCK = """

#ifdef OPENTURBO_EXPERIMENTAL_K_CACHE_PROBE
static void openturbo_probe_k_cache_write(const ggml_tensor * k_cur_3d,
                                          const ggml_tensor * k_cache,
                                          int32_t             layer_index,
                                          int64_t             kv_size,
                                          int64_t             n_stream) {
    static bool logged_once = false;
    if (logged_once || k_cur_3d == nullptr || k_cache == nullptr) {
        return;
    }

    const bool compatible = k_cur_3d->type == GGML_TYPE_F32 && k_cur_3d->ne[0] > 0 &&
                            (k_cur_3d->ne[0] % OPENTURBO_TILE_DIMS) == 0 && k_cur_3d->ne[1] > 0 &&
                            k_cur_3d->ne[2] > 0 && ggml_row_size(k_cur_3d->type, k_cur_3d->ne[0]) == k_cur_3d->nb[1];

    std::fprintf(stderr,
                 "[openturbo] cpy_k probe layer=%d compatible=%d head_dim=%lld num_heads=%lld num_tokens=%lld "
                 "num_head_tiles=%lld kv_size=%lld n_stream=%lld cache_type=%d cache_ne=[%lld,%lld,%lld]\\n",
                 layer_index, compatible ? 1 : 0, static_cast<long long>(k_cur_3d->ne[0]),
                 static_cast<long long>(k_cur_3d->ne[1]), static_cast<long long>(k_cur_3d->ne[2]),
                 static_cast<long long>(k_cur_3d->ne[0] / OPENTURBO_TILE_DIMS), static_cast<long long>(kv_size),
                 static_cast<long long>(n_stream), static_cast<int>(k_cache->type),
                 static_cast<long long>(k_cache->ne[0]), static_cast<long long>(k_cache->ne[1]),
                 static_cast<long long>(k_cache->ne[2]));

    logged_once = true;
}
#endif
"""


PROBE_CALL_BLOCK = """

#ifdef OPENTURBO_EXPERIMENTAL_K_CACHE_PROBE
    openturbo_probe_k_cache_write(k_cur, k, il, get_size(), k->ne[2]);
#endif
"""


SHADOW_EVAL_CMAKE_BLOCK = """
if (OPENTURBO_EXPERIMENTAL_K_CACHE_SHADOW_ENCODE)
    if (NOT OPENTURBO_ROOT)
        get_filename_component(OPENTURBO_ROOT_CANDIDATE "${{CMAKE_CURRENT_SOURCE_DIR}}/../.." ABSOLUTE)
        if (EXISTS "${{OPENTURBO_ROOT_CANDIDATE}}/include/openturbo/ggml_downstream.hpp")
            set(OPENTURBO_ROOT "${{OPENTURBO_ROOT_CANDIDATE}}")
        else()
            message(FATAL_ERROR "OPENTURBO_EXPERIMENTAL_K_CACHE_SHADOW_ENCODE requires OPENTURBO_ROOT to point at an OpenTurbo tree or install prefix")
        endif()
    endif()

    find_package(CUDAToolkit REQUIRED)
    find_library(OPENTURBO_C_API_LIBRARY NAMES openturbo_c_api PATHS
        "${{OPENTURBO_ROOT}}/.venv/Lib/site-packages/openturbo"
        "${{OPENTURBO_ROOT}}/build"
        NO_DEFAULT_PATH)
    find_file(OPENTURBO_C_API_DLL NAMES openturbo_c_api.dll PATHS
        "${{OPENTURBO_ROOT}}/.venv/Lib/site-packages/openturbo"
        "${{OPENTURBO_ROOT}}/build"
        NO_DEFAULT_PATH)

    if (NOT OPENTURBO_C_API_LIBRARY OR NOT OPENTURBO_C_API_DLL)
        message(FATAL_ERROR "Could not locate openturbo_c_api.lib/.dll under OPENTURBO_ROOT. Build or install OpenTurbo first.")
    endif()

    target_sources(${{TARGET}} PRIVATE {shadow_cpp_rel})
    target_include_directories(${{TARGET}} PRIVATE {shadow_dir_rel} "${{OPENTURBO_ROOT}}/include")
    target_link_libraries(${{TARGET}} PRIVATE "${{OPENTURBO_C_API_LIBRARY}}" CUDA::cudart)
    target_compile_definitions(${{TARGET}} PRIVATE OPENTURBO_EXPERIMENTAL_K_CACHE_SHADOW_ENCODE)
    add_custom_command(TARGET ${{TARGET}} POST_BUILD
        COMMAND ${{CMAKE_COMMAND}} -E copy_if_different "${{OPENTURBO_C_API_DLL}}" "$<TARGET_FILE_DIR:${{TARGET}}>")
endif()
"""


SHADOW_EVAL_INCLUDE_BLOCK = """
#ifdef OPENTURBO_EXPERIMENTAL_K_CACHE_SHADOW_ENCODE
#include \"openturbo_shadow_eval_callback.hpp\"
#endif
"""


SHADOW_EVAL_STATE_BLOCK = """
#ifdef OPENTURBO_EXPERIMENTAL_K_CACHE_SHADOW_ENCODE
    openturbo_shadow_eval_callback_state cb_data;
#else
    base_callback_data cb_data;
#endif
"""


SHADOW_EVAL_ASSIGN_BLOCK = """
    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
#ifdef OPENTURBO_EXPERIMENTAL_K_CACHE_SHADOW_ENCODE
    params.cb_eval = openturbo_shadow_cb_eval;
#else
    params.cb_eval = common_debug_cb_eval<false>;
#endif
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;
"""


def run_git_clone(target_root: Path, clone_url: str) -> None:
    subprocess.run(["git", "clone", clone_url, str(target_root)], check=True)


def write_text_file(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def patch_text_once(text: str, anchor: str, addition: str, path: Path) -> tuple[str, bool]:
    if addition in text:
        return text, False
    if anchor not in text:
        raise ValueError(f"Could not find expected anchor in {path}: {anchor!r}")
    return text.replace(anchor, anchor + addition, 1), True


def patch_probe_cmakelists(src_cmakelists: Path, output_root: Path) -> bool:
    text = src_cmakelists.read_text(encoding="utf-8")
    changed = False

    text, did_change = patch_text_once(text, "llama_add_compile_flags()\n\n", PROBE_OPTION_BLOCK, src_cmakelists)
    changed = changed or did_change

    bridge_cpp_rel = Path(os.path.relpath(output_root / "openturbo_llama_cpp_bridge.cpp", src_cmakelists.parent)).as_posix()
    bridge_dir_rel = Path(os.path.relpath(output_root, src_cmakelists.parent)).as_posix()
    probe_target_block = f"""
if (OPENTURBO_EXPERIMENTAL_K_CACHE_PROBE)
    if (NOT OPENTURBO_ROOT)
        get_filename_component(OPENTURBO_ROOT_CANDIDATE \"${{CMAKE_CURRENT_SOURCE_DIR}}/../../..\" ABSOLUTE)
        if (EXISTS \"${{OPENTURBO_ROOT_CANDIDATE}}/include/openturbo/ggml_downstream.hpp\")
            set(OPENTURBO_ROOT \"${{OPENTURBO_ROOT_CANDIDATE}}\")
        else()
            message(FATAL_ERROR \"OPENTURBO_EXPERIMENTAL_K_CACHE_PROBE requires OPENTURBO_ROOT to point at an OpenTurbo tree or install prefix\")
        endif()
    endif()

    target_sources(llama PRIVATE {bridge_cpp_rel})
    target_include_directories(llama PRIVATE {bridge_dir_rel} \"${{OPENTURBO_ROOT}}/include\")
    target_compile_definitions(llama PRIVATE OPENTURBO_EXPERIMENTAL_K_CACHE_PROBE OPENTURBO_EXPERIMENTAL_K_CACHE_PROBE_ONLY)
endif()
"""
    text, did_change = patch_text_once(text, "target_link_libraries(llama PUBLIC ggml)\n", probe_target_block, src_cmakelists)
    changed = changed or did_change

    if changed:
        src_cmakelists.write_text(text, encoding="utf-8")
    return changed


def patch_shadow_eval_callback_cmakelists(example_cmakelists: Path, output_root: Path) -> bool:
    text = example_cmakelists.read_text(encoding="utf-8")
    shadow_cpp_rel = Path(os.path.relpath(output_root / "openturbo_shadow_eval_callback.cpp", example_cmakelists.parent)).as_posix()
    shadow_dir_rel = Path(os.path.relpath(output_root, example_cmakelists.parent)).as_posix()
    block = SHADOW_EVAL_CMAKE_BLOCK.format(shadow_cpp_rel=shadow_cpp_rel, shadow_dir_rel=shadow_dir_rel)
    text, changed = patch_text_once(text, "target_compile_features(${TARGET} PRIVATE cxx_std_17)\n", block, example_cmakelists)
    if changed:
        example_cmakelists.write_text(text, encoding="utf-8")
    return changed


def patch_shadow_eval_callback_source(example_source: Path) -> bool:
    text = example_source.read_text(encoding="utf-8")
    changed = False

    text, did_change = patch_text_once(text, '#include "llama-cpp.h"\n', SHADOW_EVAL_INCLUDE_BLOCK, example_source)
    changed = changed or did_change

    if SHADOW_EVAL_STATE_BLOCK not in text:
        if "    base_callback_data cb_data;\n" not in text:
            raise ValueError(f"Could not find expected callback state anchor in {example_source}")
        text = text.replace("    base_callback_data cb_data;\n", SHADOW_EVAL_STATE_BLOCK, 1)
        changed = True

    if SHADOW_EVAL_ASSIGN_BLOCK not in text:
        anchor = "    // pass the callback to the backend scheduler\n    // it will be executed for each node during the graph computation\n    params.cb_eval = common_debug_cb_eval<false>;\n    params.cb_eval_user_data = &cb_data;\n    params.warmup = false;\n"
        if anchor not in text:
            raise ValueError(f"Could not find expected callback assignment anchor in {example_source}")
        text = text.replace(anchor, SHADOW_EVAL_ASSIGN_BLOCK, 1)
        changed = True

    if changed:
        example_source.write_text(text, encoding="utf-8")
    return changed


def patch_probe_kv_cache(src_file: Path) -> bool:
    text = src_file.read_text(encoding="utf-8")
    changed = False

    text, did_change = patch_text_once(text, '#include "llama-model.h"\n', '#include "openturbo/c_api.h"\n', src_file)
    changed = changed or did_change

    text, did_change = patch_text_once(text, '#include <cmath>\n', '#include <cstdio>\n', src_file)
    changed = changed or did_change

    text, did_change = patch_text_once(text, "static bool ggml_is_power_of_2(int n) {\n    return (n & (n - 1)) == 0;\n}\n", PROBE_HELPER_BLOCK, src_file)
    changed = changed or did_change

    text, did_change = patch_text_once(text, "    const int64_t n_embd_gqa = n_embd_head * n_head;\n", PROBE_CALL_BLOCK, src_file)
    changed = changed or did_change

    if changed:
        src_file.write_text(text, encoding="utf-8")
    return changed


def apply_probe_patch(llama_root: Path, output_root: Path) -> bool:
    src_root = llama_root / "src"
    src_cmakelists = src_root / "CMakeLists.txt"
    kv_cache_cpp = src_root / "llama-kv-cache.cpp"

    if not src_cmakelists.exists() or not kv_cache_cpp.exists():
        raise FileNotFoundError("--probe-k-cache requires a llama.cpp checkout with src/CMakeLists.txt and src/llama-kv-cache.cpp")

    changed_cmake = patch_probe_cmakelists(src_cmakelists, output_root)
    changed_kv = patch_probe_kv_cache(kv_cache_cpp)
    return changed_cmake or changed_kv


def apply_shadow_encode_patch(llama_root: Path, output_root: Path) -> bool:
    example_root = llama_root / "examples" / "eval-callback"
    example_cmakelists = example_root / "CMakeLists.txt"
    example_source = example_root / "eval-callback.cpp"

    if not example_cmakelists.exists() or not example_source.exists():
        raise FileNotFoundError("Shadow encode patch requires llama.cpp examples/eval-callback/CMakeLists.txt and eval-callback.cpp")

    changed_cmake = patch_shadow_eval_callback_cmakelists(example_cmakelists, output_root)
    changed_source = patch_shadow_eval_callback_source(example_source)
    return changed_cmake or changed_source


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an OpenTurbo integration scaffold inside a llama.cpp checkout.")
    parser.add_argument("llama_root", nargs="?", type=Path, help="Optional path to an existing llama.cpp checkout or a destination to clone into.")
    parser.add_argument("--llama-root", dest="llama_root_override", type=Path, help="Explicit llama.cpp checkout path. Overrides the positional llama_root argument when both are provided.")
    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap a local llama.cpp checkout if missing, then generate the scaffold.")
    parser.add_argument("--clone-if-missing", action="store_true", help="Clone llama.cpp into llama_root if the directory does not exist.")
    parser.add_argument("--clone-url", default=DEFAULT_LLAMA_CPP_URL, help="Repository URL used with --clone-if-missing.")
    parser.add_argument("--output-dir", type=Path, help="Explicit directory where scaffold files will be written. Overrides --output-subdir when provided.")
    parser.add_argument("--output-subdir", default="examples/openturbo", help="Subdirectory inside llama_root where the scaffold files will be written.")
    parser.add_argument("--probe-k-cache", action="store_true", help="Patch the downstream llama.cpp checkout with the experimental cpy_k probe flow.")
    parser.add_argument("--shadow-encode", action="store_true", help="Patch the downstream eval-callback example with an execution-time OpenTurbo shadow encode callback.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated files.")
    return parser


def resolve_llama_root(args: argparse.Namespace) -> Path | None:
    root_arg = args.llama_root_override or args.llama_root
    if root_arg is None:
        return (Path.cwd() / DEFAULT_LLAMA_DIR_NAME).resolve()
    return root_arg.resolve()


def ensure_llama_root_exists(llama_root: Path | None, clone_if_missing: bool, clone_url: str) -> None:
    if llama_root is None or llama_root.exists():
        return

    if clone_if_missing:
        run_git_clone(llama_root, clone_url)
        return

    llama_root.mkdir(parents=True, exist_ok=True)


def resolve_output_root(args: argparse.Namespace, llama_root: Path | None) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()

    return (llama_root / args.output_subdir).resolve()


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.bootstrap:
        args.clone_if_missing = True

    llama_root = resolve_llama_root(args)
    if args.clone_if_missing and llama_root is None:
        raise ValueError("--clone-if-missing requires a llama_root path or --llama-root.")

    ensure_llama_root_exists(llama_root, args.clone_if_missing, args.clone_url)
    output_root = resolve_output_root(args, llama_root)
    write_text_file(output_root / "openturbo_llama_cpp_bridge.hpp", BRIDGE_HEADER, args.force)
    write_text_file(output_root / "openturbo_llama_cpp_bridge.cpp", BRIDGE_CPP, args.force)
    write_text_file(output_root / "README.md", BRIDGE_README, args.force)
    write_text_file(output_root / "CMakeLists.fragment.txt", CMAKE_FRAGMENT, args.force)
    write_text_file(output_root / "openturbo_shadow_eval_callback.hpp", SHADOW_CALLBACK_HEADER, args.force)
    write_text_file(output_root / "openturbo_shadow_eval_callback.cpp", SHADOW_CALLBACK_CPP, args.force)

    patched_probe = False
    patched_shadow_encode = False
    if args.probe_k_cache:
        if llama_root is None:
            raise ValueError("--probe-k-cache requires a llama_root path or --llama-root.")
        patched_probe = apply_probe_patch(llama_root, output_root)

    if args.shadow_encode:
        if llama_root is None:
            raise ValueError("--shadow-encode requires a llama_root path or --llama-root.")
        patched_shadow_encode = apply_shadow_encode_patch(llama_root, output_root)

    print(f"Generated OpenTurbo llama.cpp scaffold in: {output_root}")
    if args.probe_k_cache:
        if patched_probe:
            print(f"Applied experimental K-cache probe patch in: {llama_root}")
        else:
            print(f"Experimental K-cache probe patch already present in: {llama_root}")
    if args.shadow_encode:
        if patched_shadow_encode:
            print(f"Applied experimental shadow-encode patch in: {llama_root}")
        else:
            print(f"Experimental shadow-encode patch already present in: {llama_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())