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

#include <cstdint>

#include \"ggml-backend.h\"

struct openturbo_shadow_eval_callback_state {
    uint8_t logged_write_layers[256] = {};
    uint8_t logged_read_layers[256] = {};
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
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {
constexpr int kOpenturboMaxTrackedLayers = 256;

struct openturbo_shadow_layer_sidecar {
    int64_t row_capacity = 0;
    int64_t tiles_per_row = 0;
    std::vector<openturbo_packed_tile_header_t> headers;
    std::vector<uint8_t> present_rows;
};

std::unordered_map<int, openturbo_shadow_layer_sidecar> g_openturbo_shadow_sidecars;

struct openturbo_layer_filter_config {
    bool initialized = false;
    bool allow_all = false;
    uint8_t enabled_layers[kOpenturboMaxTrackedLayers] = {};
};

bool openturbo_is_valid_tracked_layer(int layer_index) {
    return layer_index >= 0 && layer_index < kOpenturboMaxTrackedLayers;
}

bool openturbo_state_layer_logged(const uint8_t * flags, int layer_index) {
    return openturbo_is_valid_tracked_layer(layer_index) && flags[layer_index] != 0;
}

void openturbo_state_mark_layer_logged(uint8_t * flags, int layer_index) {
    if (openturbo_is_valid_tracked_layer(layer_index)) {
        flags[layer_index] = 1;
    }
}

const ggml_tensor * openturbo_find_named_cache_tensor(const ggml_tensor * tensor, int depth_remaining);
const ggml_tensor * openturbo_find_kq_mask_tensor(const ggml_tensor * tensor, int depth_remaining);

openturbo_layer_filter_config & openturbo_get_layer_filter_config() {
    static openturbo_layer_filter_config config;
    if (config.initialized) {
        return config;
    }

    config.initialized = true;
    const char * layer_filter = std::getenv("OPENTURBO_SHADOW_LAYER_FILTER");
    if (layer_filter == nullptr || layer_filter[0] == '\\0') {
        config.enabled_layers[0] = 1;
        return config;
    }

    if (std::strcmp(layer_filter, "all") == 0) {
        config.allow_all = true;
        return config;
    }

    const char * cursor = layer_filter;
    while (*cursor != '\\0') {
        char * end_ptr = nullptr;
        const long value = std::strtol(cursor, &end_ptr, 10);
        if (end_ptr != cursor && value >= 0 && value < kOpenturboMaxTrackedLayers) {
            config.enabled_layers[value] = 1;
            cursor = end_ptr;
        } else {
            ++cursor;
        }

        while (*cursor == ',' || *cursor == ' ' || *cursor == ';') {
            ++cursor;
        }
    }

    return config;
}

bool openturbo_should_track_layer(int layer_index) {
    if (!openturbo_is_valid_tracked_layer(layer_index)) {
        return false;
    }

    const auto & config = openturbo_get_layer_filter_config();
    return config.allow_all || config.enabled_layers[layer_index] != 0;
}

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

int openturbo_shadow_write_layer(const ggml_tensor * tensor) {
    if (!openturbo_is_shadow_candidate(tensor)) {
        return -1;
    }

    return openturbo_parse_layer_index(tensor->src[2]);
}

int openturbo_shadow_read_layer(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->op == GGML_OP_SET_ROWS) {
        return -1;
    }

    const ggml_tensor * cache_tensor = openturbo_find_named_cache_tensor(tensor, 8);
    if (cache_tensor == nullptr || openturbo_find_kq_mask_tensor(tensor, 8) == nullptr) {
        return -1;
    }

    return openturbo_parse_layer_index(cache_tensor);
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

const ggml_tensor * openturbo_find_named_cache_tensor(const ggml_tensor * tensor, int depth_remaining) {
    return openturbo_find_named_tensor_prefix(tensor, "cache_k_l", 8, depth_remaining);
}

const ggml_tensor * openturbo_find_kq_mask_tensor(const ggml_tensor * tensor, int depth_remaining) {
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

struct openturbo_shadow_score_summary {
    bool success = false;
    const char * reason = "unsupported";
    int num_heads = 0;
    int num_query_heads = 0;
    int query_head_group = 0;
    int num_query_tiles = 0;
    int active_rows = 0;
    int64_t top_row = -1;
    float top_score = 0.0f;
    int64_t first_row = -1;
    float first_score = 0.0f;
    int64_t dense_top_row = -1;
    float dense_top_score = 0.0f;
    float dense_first_score = 0.0f;
    float mean_abs_error = 0.0f;
    float max_abs_error = 0.0f;
    int top_match = 0;
    int64_t fwht_top_row = -1;
    float fwht_top_score = 0.0f;
    float fwht_first_score = 0.0f;
    float fwht_mean_abs_error = 0.0f;
    float fwht_max_abs_error = 0.0f;
    float fwht_mean_scale_ratio = 0.0f;
    float fwht_signal_power = 0.0f;
    float fwht_noise_power = 0.0f;
    float fwht_snr_db = 0.0f;
    float fwht_signal_retention = 0.0f;
    int fwht_top_match = 0;
    float component_first_main = 0.0f;
    float component_first_residual = 0.0f;
    float component_mean_main = 0.0f;
    float component_mean_residual = 0.0f;
    float legacy_first_score = 0.0f;
    float legacy_mean_abs_error = 0.0f;
    float legacy_max_abs_error = 0.0f;
    float legacy_noise_power = 0.0f;
    float legacy_snr_db = 0.0f;
    float legacy_signal_retention = 0.0f;
    int64_t legacy_top_row = -1;
    float legacy_top_score = 0.0f;
    int legacy_top_match = 0;
};

struct openturbo_scan_component_breakdown {
    float main_polar_dot = 0.0f;
    float residual_correction = 0.0f;
};

float openturbo_sign_from_bit(uint32_t bit) {
    return bit != 0 ? 1.0f : -1.0f;
}

float openturbo_compute_snr_db(float signal_power, float noise_power) {
    if (!(signal_power > 0.0f)) {
        return 0.0f;
    }

    if (!(noise_power > 0.0f)) {
        return INFINITY;
    }

    return 10.0f * std::log10(signal_power / noise_power);
}

float openturbo_compute_signal_retention(float signal_power, float noise_power) {
    const float total_power = signal_power + noise_power;
    if (!(total_power > 0.0f)) {
        return 0.0f;
    }

    return 100.0f * (signal_power / total_power);
}

void openturbo_reconstruct_pair_from_code(uint32_t code, float scale, float & x_hat, float & y_hat) {
    const uint32_t y_bit = code & 0x1u;
    const uint32_t x_bit = (code >> 1) & 0x1u;
    const float center = scale * 0.5f;
    x_hat = center * openturbo_sign_from_bit(x_bit);
    y_hat = center * openturbo_sign_from_bit(y_bit);
}

void openturbo_reconstruct_pair_from_code_corner(uint32_t code, float scale, float & x_hat, float & y_hat) {
    const uint32_t y_bit = code & 0x1u;
    const uint32_t x_bit = (code >> 1) & 0x1u;
    const float center = scale * 0.7071067811865475f;
    x_hat = center * openturbo_sign_from_bit(x_bit);
    y_hat = center * openturbo_sign_from_bit(y_bit);
}

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

bool openturbo_collect_active_rows_single_query(const ggml_tensor * cache_tensor,
                                                const ggml_tensor * mask_tensor,
                                                std::vector<int64_t> & active_rows) {
    active_rows.clear();
    if (cache_tensor == nullptr || mask_tensor == nullptr) {
        return false;
    }

    const int64_t n_kv = mask_tensor->ne[0];
    const int64_t n_tokens_per_stream = mask_tensor->ne[1];
    const int64_t n_stream = mask_tensor->ne[3];
    const int64_t rows_per_stream = openturbo_rows_per_stream(cache_tensor);
    const int64_t stream_base_index = openturbo_stream_base_index(cache_tensor);
    if (n_kv <= 0 || n_tokens_per_stream <= 0 || n_stream != 1 || rows_per_stream <= 0 || stream_base_index < 0) {
        return false;
    }

    std::vector<float> mask_values;
    if (!openturbo_copy_mask_values(mask_tensor, mask_values)) {
        return false;
    }

    active_rows.reserve(static_cast<size_t>(n_kv));
    const int64_t global_stream_base = stream_base_index * rows_per_stream;
    const size_t token_offset = static_cast<size_t>(n_tokens_per_stream - 1) * static_cast<size_t>(n_kv);
    for (int64_t row = 0; row < n_kv; ++row) {
        if (!std::isfinite(mask_values[token_offset + static_cast<size_t>(row)])) {
            continue;
        }

        const int64_t global_row = global_stream_base + row;
        active_rows.push_back(global_row);
    }

    return !active_rows.empty();
}

bool openturbo_copy_query_block_f32(const ggml_tensor * tensor,
                                    int                 num_selected_heads,
                                    int                 query_head_group,
                                    std::vector<float> & query_values) {
    query_values.clear();
    if (tensor == nullptr || tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[3] != 1) {
        return false;
    }

    const int64_t head_dim = tensor->ne[0];
    const int64_t num_heads = tensor->ne[2];
    if (num_selected_heads <= 0 || query_head_group <= 0 || num_heads < static_cast<int64_t>(num_selected_heads * query_head_group)) {
        return false;
    }

    const size_t value_count = static_cast<size_t>(head_dim) * static_cast<size_t>(num_selected_heads);
    query_values.resize(value_count);

    const size_t base_offset = static_cast<size_t>(tensor->ne[1] - 1) * static_cast<size_t>(tensor->nb[1]);
    if (tensor->type == GGML_TYPE_F32 && tensor->nb[0] == sizeof(float) && tensor->nb[2] == tensor->nb[0] * tensor->ne[0]) {
        for (int head_index = 0; head_index < num_selected_heads; ++head_index) {
            const size_t src_offset = base_offset + static_cast<size_t>(head_index * query_head_group) * static_cast<size_t>(tensor->nb[2]);
            float * const dst = query_values.data() + static_cast<size_t>(head_index) * static_cast<size_t>(head_dim);
            if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer)) {
                std::memcpy(dst, static_cast<const char *>(tensor->data) + src_offset, sizeof(float) * static_cast<size_t>(head_dim));
            } else {
                ggml_backend_tensor_get(tensor, dst, src_offset, sizeof(float) * static_cast<size_t>(head_dim));
            }
        }
        return true;
    }

    if (tensor->type == GGML_TYPE_F16 && tensor->nb[0] == sizeof(ggml_fp16_t) && tensor->nb[2] == tensor->nb[0] * tensor->ne[0]) {
        std::vector<ggml_fp16_t> raw(static_cast<size_t>(head_dim));
        for (int head_index = 0; head_index < num_selected_heads; ++head_index) {
            const size_t src_offset = base_offset + static_cast<size_t>(head_index * query_head_group) * static_cast<size_t>(tensor->nb[2]);
            if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer)) {
                std::memcpy(raw.data(), static_cast<const char *>(tensor->data) + src_offset, sizeof(ggml_fp16_t) * static_cast<size_t>(head_dim));
            } else {
                ggml_backend_tensor_get(tensor, raw.data(), src_offset, sizeof(ggml_fp16_t) * static_cast<size_t>(head_dim));
            }

            float * const dst = query_values.data() + static_cast<size_t>(head_index) * static_cast<size_t>(head_dim);
            for (size_t value_index = 0; value_index < raw.size(); ++value_index) {
                dst[value_index] = ggml_fp16_to_fp32(raw[value_index]);
            }
        }
        return true;
    }

    return false;
}

bool openturbo_copy_k_head_row_f32(const ggml_tensor * tensor,
                                   int64_t             local_row,
                                   int                 head_index,
                                   std::vector<float> & row_values) {
    row_values.clear();
    if (tensor == nullptr || tensor->ne[0] <= 0 || tensor->ne[1] <= local_row || local_row < 0 ||
        tensor->ne[2] <= head_index || head_index < 0 || tensor->ne[3] != 1) {
        return false;
    }

    const int64_t head_dim = tensor->ne[0];
    row_values.resize(static_cast<size_t>(head_dim));
    const size_t base_offset = static_cast<size_t>(local_row) * static_cast<size_t>(tensor->nb[1]) +
                               static_cast<size_t>(head_index) * static_cast<size_t>(tensor->nb[2]);

    if (tensor->type == GGML_TYPE_F32 && tensor->nb[0] == sizeof(float)) {
        if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer)) {
            std::memcpy(row_values.data(), static_cast<const char *>(tensor->data) + base_offset, sizeof(float) * static_cast<size_t>(head_dim));
        } else {
            ggml_backend_tensor_get(tensor, row_values.data(), base_offset, sizeof(float) * static_cast<size_t>(head_dim));
        }
        return true;
    }

    if (tensor->type == GGML_TYPE_F16 && tensor->nb[0] == sizeof(ggml_fp16_t)) {
        std::vector<ggml_fp16_t> raw(static_cast<size_t>(head_dim));
        if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer)) {
            std::memcpy(raw.data(), static_cast<const char *>(tensor->data) + base_offset, sizeof(ggml_fp16_t) * static_cast<size_t>(head_dim));
        } else {
            ggml_backend_tensor_get(tensor, raw.data(), base_offset, sizeof(ggml_fp16_t) * static_cast<size_t>(head_dim));
        }

        for (size_t value_index = 0; value_index < raw.size(); ++value_index) {
            row_values[value_index] = ggml_fp16_to_fp32(raw[value_index]);
        }
        return true;
    }

    return false;
}

float openturbo_dense_dot(const float * lhs, const float * rhs, int64_t count) {
    float total = 0.0f;
    for (int64_t index = 0; index < count; ++index) {
        total += lhs[index] * rhs[index];
    }
    return total;
}

void openturbo_fwht128_cpu(float * values) {
    for (int span = 1; span < OPENTURBO_TILE_DIMS; span <<= 1) {
        const int step = span << 1;
        for (int base = 0; base < OPENTURBO_TILE_DIMS; base += step) {
            for (int index = 0; index < span; ++index) {
                const float a = values[base + index];
                const float b = values[base + index + span];
                values[base + index] = a + b;
                values[base + index + span] = a - b;
            }
        }
    }
}

float openturbo_fwht_dot(const float * lhs, const float * rhs, int64_t count) {
    if (count <= 0 || (count % OPENTURBO_TILE_DIMS) != 0) {
        return 0.0f;
    }

    float total = 0.0f;
    float lhs_tile[OPENTURBO_TILE_DIMS];
    float rhs_tile[OPENTURBO_TILE_DIMS];
    for (int64_t tile_base = 0; tile_base < count; tile_base += OPENTURBO_TILE_DIMS) {
        std::memcpy(lhs_tile, lhs + tile_base, sizeof(lhs_tile));
        std::memcpy(rhs_tile, rhs + tile_base, sizeof(rhs_tile));
        openturbo_fwht128_cpu(lhs_tile);
        openturbo_fwht128_cpu(rhs_tile);
        total += openturbo_dense_dot(lhs_tile, rhs_tile, OPENTURBO_TILE_DIMS);
    }
    return total;
}

openturbo_scan_component_breakdown openturbo_estimate_scan_components(
    const openturbo_packed_tile_header_t * query_headers,
    const openturbo_packed_tile_header_t * cache_headers,
    int                                    num_query_tiles,
    bool                                   use_corner_reconstruction) {
    openturbo_scan_component_breakdown breakdown;
    for (int tile_index = 0; tile_index < num_query_tiles; ++tile_index) {
        const auto & query_header = query_headers[tile_index];
        const auto & cache_header = cache_headers[tile_index];

        const float query_scale = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(query_header.block_scale_fp16_bits));
        const float cache_scale = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(cache_header.block_scale_fp16_bits));
        const float local_alpha = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(cache_header.local_alpha_fp16_bits));

        float main_polar_dot = 0.0f;
        int qjl_correlation = 0;
        for (int pair_index = 0; pair_index < 64; ++pair_index) {
            const uint32_t q_code = pair_index < 32
                ? static_cast<uint32_t>((query_header.quadrant_word_0 >> (2 * pair_index)) & 0x3ull)
                : static_cast<uint32_t>((query_header.quadrant_word_1 >> (2 * (pair_index - 32))) & 0x3ull);
            const uint32_t k_code = pair_index < 32
                ? static_cast<uint32_t>((cache_header.quadrant_word_0 >> (2 * pair_index)) & 0x3ull)
                : static_cast<uint32_t>((cache_header.quadrant_word_1 >> (2 * (pair_index - 32))) & 0x3ull);

            float qx_hat = 0.0f;
            float qy_hat = 0.0f;
            float kx_hat = 0.0f;
            float ky_hat = 0.0f;
            if (use_corner_reconstruction) {
                openturbo_reconstruct_pair_from_code_corner(q_code, query_scale, qx_hat, qy_hat);
                openturbo_reconstruct_pair_from_code_corner(k_code, cache_scale, kx_hat, ky_hat);
            } else {
                openturbo_reconstruct_pair_from_code(q_code, query_scale, qx_hat, qy_hat);
                openturbo_reconstruct_pair_from_code(k_code, cache_scale, kx_hat, ky_hat);
            }
            main_polar_dot += qx_hat * kx_hat + qy_hat * ky_hat;

            const int q_bit = ((query_header.qjl_sign_word >> pair_index) & 0x1ull) ? 1 : -1;
            const int k_bit = ((cache_header.qjl_sign_word >> pair_index) & 0x1ull) ? 1 : -1;
            qjl_correlation += q_bit * k_bit;
        }

        breakdown.main_polar_dot += main_polar_dot;
        breakdown.residual_correction += local_alpha * static_cast<float>(qjl_correlation);
    }

    return breakdown;
}

openturbo_shadow_score_summary openturbo_run_shadow_score(const ggml_tensor *                    q_tensor,
                                                          const ggml_tensor *                    k_tensor,
                                                          const ggml_tensor *                    cache_tensor,
                                                          const ggml_tensor *                    mask_tensor,
                                                          const openturbo_shadow_layer_sidecar & sidecar) {
    openturbo_shadow_score_summary summary;
    if (q_tensor == nullptr || k_tensor == nullptr || cache_tensor == nullptr || mask_tensor == nullptr) {
        summary.reason = "missing_tensor";
        return summary;
    }

    std::vector<int64_t> active_rows;
    if (!openturbo_collect_active_rows_single_query(cache_tensor, mask_tensor, active_rows)) {
        summary.reason = "unsupported_mask_shape";
        return summary;
    }

    const int64_t head_dim = q_tensor->ne[0];
    if (head_dim <= 0 || (head_dim % OPENTURBO_TILE_DIMS) != 0) {
        summary.reason = "unsupported_head_dim";
        return summary;
    }

    const int num_query_tiles = static_cast<int>(head_dim / OPENTURBO_TILE_DIMS);
    const int num_query_heads = static_cast<int>(q_tensor->ne[2]);
    const int num_heads = static_cast<int>(sidecar.tiles_per_row / num_query_tiles);
    if (num_query_heads <= 0 || num_heads <= 0 || sidecar.tiles_per_row != static_cast<int64_t>(num_query_tiles) * num_heads) {
        summary.reason = "sidecar_shape_mismatch";
        return summary;
    }

    if (num_query_heads % num_heads != 0) {
        summary.reason = "unsupported_gqa_ratio";
        return summary;
    }

    const int query_head_group = num_query_heads / num_heads;
    std::vector<float> query_values;
    if (!openturbo_copy_query_block_f32(q_tensor, num_query_heads, 1, query_values)) {
        summary.reason = "unsupported_query_layout";
        return summary;
    }

    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count <= 0) {
        summary.reason = "no_cuda_device";
        return summary;
    }

    std::vector<openturbo_packed_tile_header_t> host_cache_headers(
        static_cast<size_t>(num_heads) * active_rows.size() * static_cast<size_t>(num_query_tiles));
    for (size_t active_index = 0; active_index < active_rows.size(); ++active_index) {
        const int64_t row_index = active_rows[active_index];
        if (row_index < 0 || row_index >= sidecar.row_capacity) {
            summary.reason = "active_row_oob";
            return summary;
        }

        const size_t row_offset = static_cast<size_t>(row_index) * static_cast<size_t>(sidecar.tiles_per_row);
        for (int head_index = 0; head_index < num_heads; ++head_index) {
            const size_t src_offset = row_offset + static_cast<size_t>(head_index * num_query_tiles);
            const size_t dst_offset = (static_cast<size_t>(head_index) * active_rows.size() + active_index) *
                                      static_cast<size_t>(num_query_tiles);
            std::copy_n(sidecar.headers.data() + src_offset,
                        static_cast<size_t>(num_query_tiles),
                        host_cache_headers.data() + dst_offset);
        }
    }

    float * device_query_input = nullptr;
    openturbo_packed_tile_header_t * device_query_headers = nullptr;
    openturbo_packed_tile_header_t * device_cache_headers = nullptr;
    float * device_scores = nullptr;

    const size_t query_input_bytes = sizeof(float) * query_values.size();
    const size_t query_header_count = static_cast<size_t>(num_query_heads) * static_cast<size_t>(num_query_tiles);
    const size_t cache_header_count = host_cache_headers.size();
    const size_t score_count = static_cast<size_t>(num_query_heads) * active_rows.size();

    cuda_status = cudaMalloc(&device_query_input, query_input_bytes);
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaMalloc(&device_query_headers, sizeof(openturbo_packed_tile_header_t) * query_header_count);
    }
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaMalloc(&device_cache_headers, sizeof(openturbo_packed_tile_header_t) * cache_header_count);
    }
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaMalloc(&device_scores, sizeof(float) * score_count);
    }
    if (cuda_status != cudaSuccess) {
        summary.reason = "cuda_alloc_failed";
        goto cleanup;
    }

    cuda_status = cudaMemcpy(device_query_input, query_values.data(), query_input_bytes, cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaMemcpy(device_cache_headers,
                                 host_cache_headers.data(),
                                 sizeof(openturbo_packed_tile_header_t) * cache_header_count,
                                 cudaMemcpyHostToDevice);
    }
    if (cuda_status != cudaSuccess) {
        summary.reason = "cuda_memcpy_failed";
        goto cleanup;
    }

    {
        int openturbo_cuda_status = 0;
        const openturbo_status_t encode_status = openturbo_encode_tile_fused_prerotated(
            device_query_input,
            device_query_headers,
            static_cast<int>(query_header_count),
            nullptr,
            &openturbo_cuda_status);
        if (encode_status != OPENTURBO_STATUS_SUCCESS) {
            summary.reason = "query_encode_failed";
            goto cleanup;
        }

        for (int query_head_index = 0; query_head_index < num_query_heads; ++query_head_index) {
            const int kv_head_index = query_head_index / query_head_group;
            const openturbo_status_t scan_status = openturbo_scan_query_many_cache_multi_tile(
                device_query_headers + static_cast<size_t>(query_head_index * num_query_tiles),
                device_cache_headers + static_cast<size_t>(kv_head_index) * active_rows.size() * static_cast<size_t>(num_query_tiles),
                device_scores + static_cast<size_t>(query_head_index) * active_rows.size(),
                num_query_tiles,
                static_cast<int>(active_rows.size()),
                nullptr,
                &openturbo_cuda_status);
            if (scan_status != OPENTURBO_STATUS_SUCCESS) {
                summary.reason = "cache_scan_failed";
                goto cleanup;
            }
        }
    }

    {
        std::vector<float> host_scores(score_count);
        std::vector<openturbo_packed_tile_header_t> host_query_headers(query_header_count);
        cuda_status = cudaMemcpy(host_scores.data(), device_scores, sizeof(float) * score_count, cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            summary.reason = "score_copy_failed";
            goto cleanup;
        }

        cuda_status = cudaMemcpy(host_query_headers.data(),
                                 device_query_headers,
                                 sizeof(openturbo_packed_tile_header_t) * query_header_count,
                                 cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            summary.reason = "query_header_copy_failed";
            goto cleanup;
        }

        summary.success = true;
        summary.reason = "success";
        summary.num_heads = num_heads;
        summary.num_query_heads = num_query_heads;
        summary.query_head_group = query_head_group;
        summary.num_query_tiles = num_query_tiles;
        summary.active_rows = static_cast<int>(active_rows.size());
        summary.first_row = active_rows.front();

        float best_score = -INFINITY;
        float best_dense_score = -INFINITY;
        float best_fwht_score = -INFINITY;
        float best_legacy_score = -INFINITY;
        float total_abs_error = 0.0f;
        float total_fwht_abs_error = 0.0f;
        float total_fwht_scale_ratio = 0.0f;
        float total_fwht_signal_sq = 0.0f;
        float total_fwht_noise_sq = 0.0f;
        float total_legacy_abs_error = 0.0f;
        float total_legacy_noise_sq = 0.0f;
        float total_component_main = 0.0f;
        float total_component_residual = 0.0f;
        int fwht_scale_ratio_count = 0;
        const int64_t rows_per_stream = openturbo_rows_per_stream(cache_tensor);
        const int64_t stream_base_index = openturbo_stream_base_index(cache_tensor);
        const int64_t local_row_base = stream_base_index * rows_per_stream;
        std::vector<float> k_row_values;
        for (size_t active_index = 0; active_index < active_rows.size(); ++active_index) {
            float mean_score = 0.0f;
            float mean_dense_score = 0.0f;
            float mean_fwht_score = 0.0f;
            float mean_component_main = 0.0f;
            float mean_component_residual = 0.0f;
            float mean_legacy_main = 0.0f;
            float mean_legacy_residual = 0.0f;
            const int64_t local_row = active_rows[active_index] - local_row_base;
            for (int query_head_index = 0; query_head_index < num_query_heads; ++query_head_index) {
                const int kv_head_index = query_head_index / query_head_group;
                const size_t score_offset = static_cast<size_t>(query_head_index) * active_rows.size() + active_index;
                const size_t cache_offset = (static_cast<size_t>(kv_head_index) * active_rows.size() + active_index) * static_cast<size_t>(num_query_tiles);
                mean_score += host_scores[score_offset];

                const auto component_breakdown = openturbo_estimate_scan_components(
                    host_query_headers.data() + static_cast<size_t>(query_head_index * num_query_tiles),
                    host_cache_headers.data() + cache_offset,
                    num_query_tiles,
                    false);
                const auto legacy_breakdown = openturbo_estimate_scan_components(
                    host_query_headers.data() + static_cast<size_t>(query_head_index * num_query_tiles),
                    host_cache_headers.data() + cache_offset,
                    num_query_tiles,
                    true);
                total_component_main += component_breakdown.main_polar_dot;
                total_component_residual += component_breakdown.residual_correction;
                mean_component_main += component_breakdown.main_polar_dot;
                mean_component_residual += component_breakdown.residual_correction;
                mean_legacy_main += legacy_breakdown.main_polar_dot;
                mean_legacy_residual += legacy_breakdown.residual_correction;

                if (!openturbo_copy_k_head_row_f32(k_tensor, local_row, kv_head_index, k_row_values)) {
                    summary.reason = "unsupported_k_layout";
                    summary.success = false;
                    goto cleanup;
                }

                const float * const query_head = query_values.data() + static_cast<size_t>(query_head_index) * static_cast<size_t>(head_dim);
                mean_dense_score += openturbo_dense_dot(query_head, k_row_values.data(), head_dim);
                mean_fwht_score += openturbo_fwht_dot(query_head, k_row_values.data(), head_dim);
            }
            mean_score /= static_cast<float>(num_query_heads);
            mean_dense_score /= static_cast<float>(num_query_heads);
            mean_fwht_score /= static_cast<float>(num_query_heads);
            mean_component_main /= static_cast<float>(num_query_heads);
            mean_component_residual /= static_cast<float>(num_query_heads);
            mean_legacy_main /= static_cast<float>(num_query_heads);
            mean_legacy_residual /= static_cast<float>(num_query_heads);
            const float mean_legacy_score = mean_legacy_main + mean_legacy_residual;
            const float abs_error = std::fabs(mean_score - mean_dense_score);
            const float fwht_abs_error = std::fabs(mean_score - mean_fwht_score);
            const float legacy_abs_error = std::fabs(mean_legacy_score - mean_fwht_score);
            const float fwht_noise = mean_score - mean_fwht_score;
            const float legacy_noise = mean_legacy_score - mean_fwht_score;
            total_abs_error += abs_error;
            total_fwht_abs_error += fwht_abs_error;
            total_legacy_abs_error += legacy_abs_error;
            total_fwht_signal_sq += mean_fwht_score * mean_fwht_score;
            total_fwht_noise_sq += fwht_noise * fwht_noise;
            total_legacy_noise_sq += legacy_noise * legacy_noise;
            summary.max_abs_error = std::max(summary.max_abs_error, abs_error);
            summary.fwht_max_abs_error = std::max(summary.fwht_max_abs_error, fwht_abs_error);
            summary.legacy_max_abs_error = std::max(summary.legacy_max_abs_error, legacy_abs_error);
            if (std::fabs(mean_fwht_score) > 1.0e-6f) {
                total_fwht_scale_ratio += mean_score / mean_fwht_score;
                ++fwht_scale_ratio_count;
            }

            if (active_index == 0) {
                summary.first_score = mean_score;
                summary.dense_first_score = mean_dense_score;
                summary.fwht_first_score = mean_fwht_score;
                summary.component_first_main = mean_component_main;
                summary.component_first_residual = mean_component_residual;
                summary.legacy_first_score = mean_legacy_score;
            }
            if (mean_score > best_score) {
                best_score = mean_score;
                summary.top_score = mean_score;
                summary.top_row = active_rows[active_index];
            }
            if (mean_dense_score > best_dense_score) {
                best_dense_score = mean_dense_score;
                summary.dense_top_score = mean_dense_score;
                summary.dense_top_row = active_rows[active_index];
            }
            if (mean_fwht_score > best_fwht_score) {
                best_fwht_score = mean_fwht_score;
                summary.fwht_top_score = mean_fwht_score;
                summary.fwht_top_row = active_rows[active_index];
            }
            if (mean_legacy_score > best_legacy_score) {
                best_legacy_score = mean_legacy_score;
                summary.legacy_top_score = mean_legacy_score;
                summary.legacy_top_row = active_rows[active_index];
            }
        }

        summary.mean_abs_error = total_abs_error / static_cast<float>(active_rows.size());
        summary.top_match = summary.top_row == summary.dense_top_row ? 1 : 0;
        summary.fwht_mean_abs_error = total_fwht_abs_error / static_cast<float>(active_rows.size());
        summary.fwht_top_match = summary.top_row == summary.fwht_top_row ? 1 : 0;
        summary.fwht_mean_scale_ratio = fwht_scale_ratio_count > 0 ? total_fwht_scale_ratio / static_cast<float>(fwht_scale_ratio_count) : 0.0f;
        summary.fwht_signal_power = total_fwht_signal_sq / static_cast<float>(active_rows.size());
        summary.fwht_noise_power = total_fwht_noise_sq / static_cast<float>(active_rows.size());
        summary.fwht_snr_db = openturbo_compute_snr_db(summary.fwht_signal_power, summary.fwht_noise_power);
        summary.fwht_signal_retention = openturbo_compute_signal_retention(summary.fwht_signal_power, summary.fwht_noise_power);
        summary.component_mean_main = total_component_main / static_cast<float>(active_rows.size() * num_query_heads);
        summary.component_mean_residual = total_component_residual / static_cast<float>(active_rows.size() * num_query_heads);
        summary.legacy_mean_abs_error = total_legacy_abs_error / static_cast<float>(active_rows.size());
        summary.legacy_noise_power = total_legacy_noise_sq / static_cast<float>(active_rows.size());
        summary.legacy_snr_db = openturbo_compute_snr_db(summary.fwht_signal_power, summary.legacy_noise_power);
        summary.legacy_signal_retention = openturbo_compute_signal_retention(summary.fwht_signal_power, summary.legacy_noise_power);
        summary.legacy_top_match = summary.legacy_top_row == summary.fwht_top_row ? 1 : 0;
    }

cleanup:
    if (device_scores != nullptr) {
        cudaFree(device_scores);
    }
    if (device_cache_headers != nullptr) {
        cudaFree(device_cache_headers);
    }
    if (device_query_headers != nullptr) {
        cudaFree(device_query_headers);
    }
    if (device_query_input != nullptr) {
        cudaFree(device_query_input);
    }
    return summary;
}

bool openturbo_is_read_candidate(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }

    if (tensor->op == GGML_OP_SET_ROWS) {
        return false;
    }

    return openturbo_find_named_cache_tensor(tensor, 8) != nullptr &&
           openturbo_find_kq_mask_tensor(tensor, 8) != nullptr;
}
}

bool openturbo_shadow_cb_eval(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto * state = static_cast<openturbo_shadow_eval_callback_state *>(user_data);

    if (ask) {
        const int write_layer = openturbo_shadow_write_layer(tensor);
        const int read_layer = openturbo_shadow_read_layer(tensor);
        const bool wants_write = openturbo_should_track_layer(write_layer) &&
            (state == nullptr || !openturbo_state_layer_logged(state->logged_write_layers, write_layer));
        const bool wants_read = openturbo_should_track_layer(read_layer) &&
            (state == nullptr || !openturbo_state_layer_logged(state->logged_read_layers, read_layer));
        return wants_write || wants_read;
    }

    if (openturbo_is_read_candidate(tensor)) {
        const int tracked_layer = openturbo_shadow_read_layer(tensor);
        if (!openturbo_should_track_layer(tracked_layer)) {
            return true;
        }

        if (state != nullptr && openturbo_state_layer_logged(state->logged_read_layers, tracked_layer)) {
            return true;
        }

        const ggml_tensor * cache_tensor = openturbo_find_named_cache_tensor(tensor, 8);
        const ggml_tensor * mask_tensor = openturbo_find_kq_mask_tensor(tensor, 8);
        const int layer_index = openturbo_parse_layer_index(cache_tensor);
        const auto sidecar_it = g_openturbo_shadow_sidecars.find(layer_index);
        if (sidecar_it == g_openturbo_shadow_sidecars.end()) {
            std::fprintf(stderr,
                         \"[openturbo] shadow_read layer=%d status=missing_sidecar node=%s\\n\",
                         layer_index,
                         ggml_get_name(tensor));
            if (state != nullptr) {
                openturbo_state_mark_layer_logged(state->logged_read_layers, layer_index);
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
                openturbo_state_mark_layer_logged(state->logged_read_layers, layer_index);
                state->logged_error = true;
            }
            return true;
        }

        std::fprintf(stderr,
                     "[openturbo] shadow_read layer=%d status=%s node=%s expected_rows=%lld present_rows=%d tiles_per_row=%lld\\n",
                     layer_index,
                     coverage.present_rows == coverage.expected_rows ? "success" : "partial",
                     ggml_get_name(tensor),
                     static_cast<long long>(coverage.expected_rows),
                     coverage.present_rows,
                     static_cast<long long>(sidecar.tiles_per_row));

        if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            const auto score_summary = openturbo_run_shadow_score(tensor->src[0], tensor->src[1], cache_tensor, mask_tensor, sidecar);
            if (score_summary.success) {
                std::fprintf(stderr,
                             "[openturbo] shadow_score layer=%d status=success node=%s active_rows=%d num_heads=%d num_query_heads=%d query_head_group=%d num_query_tiles=%d first_row=%lld first_score=%.6f top_row=%lld top_score=%.6f\\n",
                             layer_index,
                             ggml_get_name(tensor),
                             score_summary.active_rows,
                             score_summary.num_heads,
                             score_summary.num_query_heads,
                             score_summary.query_head_group,
                             score_summary.num_query_tiles,
                             static_cast<long long>(score_summary.first_row),
                             score_summary.first_score,
                             static_cast<long long>(score_summary.top_row),
                             score_summary.top_score);
                std::fprintf(stderr,
                             "[openturbo] shadow_compare layer=%d status=success node=%s active_rows=%d raw_top_match=%d fwht_top_match=%d first_row=%lld shadow_first=%.6f raw_first=%.6f fwht_first=%.6f shadow_top_row=%lld shadow_top=%.6f raw_top_row=%lld raw_top=%.6f fwht_top_row=%lld fwht_top=%.6f raw_mae=%.6f raw_max_abs_error=%.6f fwht_mae=%.6f fwht_max_abs_error=%.6f fwht_mean_scale_ratio=%.6f\\n",
                             layer_index,
                             ggml_get_name(tensor),
                             score_summary.active_rows,
                             score_summary.top_match,
                             score_summary.fwht_top_match,
                             static_cast<long long>(score_summary.first_row),
                             score_summary.first_score,
                             score_summary.dense_first_score,
                             score_summary.fwht_first_score,
                             static_cast<long long>(score_summary.top_row),
                             score_summary.top_score,
                             static_cast<long long>(score_summary.dense_top_row),
                             score_summary.dense_top_score,
                             static_cast<long long>(score_summary.fwht_top_row),
                             score_summary.fwht_top_score,
                             score_summary.mean_abs_error,
                             score_summary.max_abs_error,
                             score_summary.fwht_mean_abs_error,
                             score_summary.fwht_max_abs_error,
                             score_summary.fwht_mean_scale_ratio);
                std::fprintf(stderr,
                             "[openturbo] shadow_snr layer=%d status=success node=%s fwht_signal_power=%.6f fwht_noise_power=%.6f fwht_snr_db=%.6f signal_retention=%.6f legacy_noise_power=%.6f legacy_snr_db=%.6f legacy_signal_retention=%.6f\\n",
                             layer_index,
                             ggml_get_name(tensor),
                             score_summary.fwht_signal_power,
                             score_summary.fwht_noise_power,
                             score_summary.fwht_snr_db,
                             score_summary.fwht_signal_retention,
                             score_summary.legacy_noise_power,
                             score_summary.legacy_snr_db,
                             score_summary.legacy_signal_retention);
                std::fprintf(stderr,
                             "[openturbo] shadow_components layer=%d status=success node=%s first_main=%.6f first_residual=%.6f mean_main=%.6f mean_residual=%.6f\\n",
                             layer_index,
                             ggml_get_name(tensor),
                             score_summary.component_first_main,
                             score_summary.component_first_residual,
                             score_summary.component_mean_main,
                             score_summary.component_mean_residual);
#ifdef OPENTURBO_EXPERIMENTAL_PACKED_SCORE_PATH
                std::fprintf(stderr,
                             "[openturbo] shadow_packed_path layer=%d status=prepared node=%s active_rows=%d shadow_top_row=%lld shadow_top=%.6f fwht_top_match=%d\\n",
                             layer_index,
                             ggml_get_name(tensor),
                             score_summary.active_rows,
                             static_cast<long long>(score_summary.top_row),
                             score_summary.top_score,
                             score_summary.fwht_top_match);
#endif
                std::fprintf(stderr,
                             "[openturbo] shadow_legacy layer=%d status=success node=%s corner_top_match=%d corner_first=%.6f corner_top_row=%lld corner_top=%.6f corner_mae=%.6f corner_max_abs_error=%.6f\\n",
                             layer_index,
                             ggml_get_name(tensor),
                             score_summary.legacy_top_match,
                             score_summary.legacy_first_score,
                             static_cast<long long>(score_summary.legacy_top_row),
                             score_summary.legacy_top_score,
                             score_summary.legacy_mean_abs_error,
                             score_summary.legacy_max_abs_error);
            } else {
                std::fprintf(stderr,
                             "[openturbo] shadow_score layer=%d status=%s node=%s\\n",
                             layer_index,
                             score_summary.reason,
                             ggml_get_name(tensor));
                std::fprintf(stderr,
                             "[openturbo] shadow_compare layer=%d status=%s node=%s\\n",
                             layer_index,
                             score_summary.reason,
                             ggml_get_name(tensor));
                std::fprintf(stderr,
                             "[openturbo] shadow_snr layer=%d status=%s node=%s\\n",
                             layer_index,
                             score_summary.reason,
                             ggml_get_name(tensor));
                std::fprintf(stderr,
                             "[openturbo] shadow_components layer=%d status=%s node=%s\\n",
                             layer_index,
                             score_summary.reason,
                             ggml_get_name(tensor));
#ifdef OPENTURBO_EXPERIMENTAL_PACKED_SCORE_PATH
                std::fprintf(stderr,
                             "[openturbo] shadow_packed_path layer=%d status=%s node=%s\\n",
                             layer_index,
                             score_summary.reason,
                             ggml_get_name(tensor));
#endif
                std::fprintf(stderr,
                             "[openturbo] shadow_legacy layer=%d status=%s node=%s\\n",
                             layer_index,
                             score_summary.reason,
                             ggml_get_name(tensor));
            }
        }
        if (state != nullptr) {
            openturbo_state_mark_layer_logged(state->logged_read_layers, layer_index);
        }
        return true;
    }

    if (!openturbo_is_shadow_candidate(tensor)) {
        return true;
    }

    const int tracked_layer = openturbo_shadow_write_layer(tensor);
    if (!openturbo_should_track_layer(tracked_layer)) {
        return true;
    }

    if (state != nullptr && openturbo_state_layer_logged(state->logged_write_layers, tracked_layer)) {
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
            openturbo_state_mark_layer_logged(state->logged_write_layers, layer_index);
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
option(OPENTURBO_EXPERIMENTAL_PACKED_SCORE_PATH \"llama: prepare packed score path plumbing in the eval callback example\" OFF)
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
    if (OPENTURBO_EXPERIMENTAL_PACKED_SCORE_PATH)
        target_compile_definitions(${{TARGET}} PRIVATE OPENTURBO_EXPERIMENTAL_PACKED_SCORE_PATH)
    endif()
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