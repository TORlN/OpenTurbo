from __future__ import annotations

import argparse
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
"""


BRIDGE_CPP = """#include \"openturbo_llama_cpp_bridge.hpp\"

openturbo_status_t openturbo_llama_cpp_encode(
    const openturbo_llama_cpp_encode_request * request,
    int * cuda_status_out) {
    if (request == nullptr || request->input_heads == nullptr || request->output_headers_by_head == nullptr) {
        return OPENTURBO_STATUS_INVALID_ARGUMENT;
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
"""


BRIDGE_README = """# OpenTurbo llama.cpp Scaffold

This scaffold is generated from OpenTurbo and is intended to live inside a llama.cpp checkout.

What it gives you:

- `openturbo_llama_cpp_bridge.hpp`: a small downstream-facing request API.
- `openturbo_llama_cpp_bridge.cpp`: wrappers that forward real `ggml_tensor` objects into OpenTurbo.
- `CMakeLists.fragment.txt`: a minimal build fragment showing where to add the bridge source.

What you still need to do in llama.cpp:

1. Identify the K/V cache tensors and head-local tile counts at the actual call site.
2. Fill the request structs with those tensors and the right `num_query_tiles` / `num_cache_tokens` values.
3. Link against the installed OpenTurbo headers and `openturbo_c_api` library.

This scaffold does not attempt to patch llama.cpp automatically because the exact call site and build layout vary across revisions.
"""


CMAKE_FRAGMENT = """# Add this source inside the downstream llama.cpp build after OpenTurbo is available.
# Example:
# target_sources(llama PRIVATE path/to/openturbo_llama_cpp_bridge.cpp)
# target_include_directories(llama PRIVATE <OpenTurbo install prefix>/include)
# target_link_libraries(llama PRIVATE openturbo_c_api)
"""


def run_git_clone(target_root: Path, clone_url: str) -> None:
    subprocess.run(["git", "clone", clone_url, str(target_root)], check=True)


def write_text_file(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an OpenTurbo integration scaffold inside a llama.cpp checkout.")
    parser.add_argument("llama_root", nargs="?", type=Path, help="Optional path to an existing llama.cpp checkout or a destination to clone into.")
    parser.add_argument("--llama-root", dest="llama_root_override", type=Path, help="Explicit llama.cpp checkout path. Overrides the positional llama_root argument when both are provided.")
    parser.add_argument("--clone-if-missing", action="store_true", help="Clone llama.cpp into llama_root if the directory does not exist.")
    parser.add_argument("--clone-url", default=DEFAULT_LLAMA_CPP_URL, help="Repository URL used with --clone-if-missing.")
    parser.add_argument("--output-dir", type=Path, help="Explicit directory where scaffold files will be written. Overrides --output-subdir when provided.")
    parser.add_argument("--output-subdir", default="examples/openturbo", help="Subdirectory inside llama_root where the scaffold files will be written.")
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
    llama_root = resolve_llama_root(args)
    if args.clone_if_missing and llama_root is None:
        raise ValueError("--clone-if-missing requires a llama_root path or --llama-root.")

    ensure_llama_root_exists(llama_root, args.clone_if_missing, args.clone_url)
    output_root = resolve_output_root(args, llama_root)
    write_text_file(output_root / "openturbo_llama_cpp_bridge.hpp", BRIDGE_HEADER, args.force)
    write_text_file(output_root / "openturbo_llama_cpp_bridge.cpp", BRIDGE_CPP, args.force)
    write_text_file(output_root / "README.md", BRIDGE_README, args.force)
    write_text_file(output_root / "CMakeLists.fragment.txt", CMAKE_FRAGMENT, args.force)

    print(f"Generated OpenTurbo llama.cpp scaffold in: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())