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

    patched_probe = False
    if args.probe_k_cache:
        if llama_root is None:
            raise ValueError("--probe-k-cache requires a llama_root path or --llama-root.")
        patched_probe = apply_probe_patch(llama_root, output_root)

    print(f"Generated OpenTurbo llama.cpp scaffold in: {output_root}")
    if args.probe_k_cache:
        if patched_probe:
            print(f"Applied experimental K-cache probe patch in: {llama_root}")
        else:
            print(f"Experimental K-cache probe patch already present in: {llama_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())