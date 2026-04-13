from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from scaffold_llama_cpp_integration import apply_probe_patch
from scaffold_llama_cpp_integration import ensure_llama_root_exists
from scaffold_llama_cpp_integration import resolve_output_root
from scaffold_llama_cpp_integration import resolve_llama_root
from scaffold_llama_cpp_integration import write_text_file
from scaffold_llama_cpp_integration import BRIDGE_CPP
from scaffold_llama_cpp_integration import BRIDGE_HEADER
from scaffold_llama_cpp_integration import BRIDGE_README
from scaffold_llama_cpp_integration import CMAKE_FRAGMENT
from scaffold_llama_cpp_integration import DEFAULT_LLAMA_CPP_URL


MODEL_NAME = "tinyllamas/stories15M-q4_0.gguf"
MODEL_HASH = "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739"
PROBE_PATTERN = re.compile(r"^\[openturbo\] cpy_k probe .*$", re.MULTILINE)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap llama.cpp, apply the OpenTurbo K-cache probe patch, build it, fetch a tiny model, and run one probe inference."
    )
    parser.add_argument(
        "llama_root",
        nargs="?",
        type=Path,
        help="Optional path to the downstream llama.cpp checkout. Defaults to ./llama.",
    )
    parser.add_argument(
        "--llama-root",
        dest="llama_root_override",
        type=Path,
        help="Explicit llama.cpp checkout path. Overrides the positional llama_root argument.",
    )
    parser.add_argument(
        "--clone-url",
        default=DEFAULT_LLAMA_CPP_URL,
        help="Repository URL to clone when the checkout is missing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit directory where scaffold files will be written. Defaults to <llama_root>/examples/openturbo.",
    )
    parser.add_argument(
        "--output-subdir",
        default="examples/openturbo",
        help="Subdirectory inside llama_root where the scaffold files will be written when --output-dir is omitted.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Explicit CMake build directory. Defaults to <llama_root>/build-probe.",
    )
    parser.add_argument(
        "--config",
        default="Release",
        help="CMake build configuration to use. Default: Release.",
    )
    parser.add_argument(
        "--target",
        default="llama-eval-callback",
        help="CMake target to build and run for the probe. Default: llama-eval-callback.",
    )
    parser.add_argument(
        "--prompt",
        default="hello",
        help="Prompt passed to the eval callback runner. Default: hello.",
    )
    parser.add_argument(
        "--seed",
        default="42",
        help="Seed passed to the eval callback runner. Default: 42.",
    )
    parser.add_argument(
        "--ngl",
        default="0",
        help="GPU layers argument passed to the runner. Default: 0.",
    )
    parser.add_argument(
        "--openturbo-root",
        type=Path,
        help="Explicit OpenTurbo root to pass through CMake. Defaults to the current repo root.",
    )
    parser.add_argument(
        "--cmake",
        type=Path,
        help="Explicit cmake executable path. Defaults to the first cmake on PATH.",
    )
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="Refuse to overwrite existing generated scaffold files.",
    )
    return parser


def run_command(command: list[str], cwd: Path | None = None, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=None if cwd is None else str(cwd),
        check=True,
        text=True,
        capture_output=capture_output,
    )


def resolve_cmake_executable(explicit_cmake: Path | None) -> str:
    if explicit_cmake is not None:
        return str(explicit_cmake)

    cmake_path = shutil.which("cmake")
    if cmake_path:
        return cmake_path

    raise FileNotFoundError("Could not find 'cmake' on PATH. Pass --cmake with an explicit executable path.")


def generate_scaffold(output_root: Path, force: bool) -> None:
    write_text_file(output_root / "openturbo_llama_cpp_bridge.hpp", BRIDGE_HEADER, force)
    write_text_file(output_root / "openturbo_llama_cpp_bridge.cpp", BRIDGE_CPP, force)
    write_text_file(output_root / "README.md", BRIDGE_README, force)
    write_text_file(output_root / "CMakeLists.fragment.txt", CMAKE_FRAGMENT, force)


def configure_probe_build(cmake_exe: str, llama_root: Path, build_dir: Path, config: str, openturbo_root: Path) -> None:
    run_command(
        [
            cmake_exe,
            "-S",
            str(llama_root),
            "-B",
            str(build_dir),
            f"-DOPENTURBO_EXPERIMENTAL_K_CACHE_PROBE=ON",
            f"-DOPENTURBO_ROOT={openturbo_root}",
        ]
    )


def build_probe_target(cmake_exe: str, build_dir: Path, config: str, target: str) -> None:
    run_command([cmake_exe, "--build", str(build_dir), "--config", config, "--target", target])


def download_tiny_model(cmake_exe: str, llama_root: Path, build_dir: Path) -> Path:
    model_dest = build_dir / MODEL_NAME.replace("/", os.sep)
    download_script = llama_root / "cmake" / "download-models.cmake"
    run_command(
        [
            cmake_exe,
            f"-DDEST={model_dest.as_posix()}",
            f"-DNAME={MODEL_NAME}",
            f"-DHASH={MODEL_HASH}",
            "-P",
            str(download_script),
        ]
    )
    return model_dest


def resolve_runner(build_dir: Path, config: str, target: str) -> Path:
    exe_name = target if target.lower().endswith(".exe") else f"{target}.exe"
    return build_dir / "bin" / config / exe_name


def run_probe(runner: Path, model_path: Path, prompt: str, seed: str, ngl: str) -> str:
    completed = subprocess.run(
        [
            str(runner),
            "-m",
            str(model_path),
            "--prompt",
            prompt,
            "--seed",
            seed,
            "-ngl",
            ngl,
        ],
        text=True,
        capture_output=True,
    )

    output = (completed.stdout or "") + (completed.stderr or "")
    match = PROBE_PATTERN.search(output)
    if match is None:
        raise RuntimeError(
            "Probe run completed without an OpenTurbo cpy_k probe line. "
            f"Exit code was {completed.returncode}."
        )

    return match.group(0)


def main() -> int:
    args = build_arg_parser().parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cmake_exe = resolve_cmake_executable(args.cmake)
    openturbo_root = (args.openturbo_root or repo_root).resolve()

    llama_root = resolve_llama_root(args)
    if llama_root is None:
        raise ValueError("A llama.cpp checkout path is required.")
    ensure_llama_root_exists(llama_root, clone_if_missing=True, clone_url=args.clone_url)
    output_root = resolve_output_root(args, llama_root)
    build_dir = (args.build_dir or (llama_root / "build-probe")).resolve()

    generate_scaffold(output_root, not args.no_force)
    apply_probe_patch(llama_root, output_root)
    configure_probe_build(cmake_exe, llama_root, build_dir, args.config, openturbo_root)
    build_probe_target(cmake_exe, build_dir, args.config, args.target)
    model_path = download_tiny_model(cmake_exe, llama_root, build_dir)
    runner = resolve_runner(build_dir, args.config, args.target)
    probe_line = run_probe(runner, model_path, args.prompt, args.seed, args.ngl)

    print(f"llama_root={llama_root}")
    print(f"build_dir={build_dir}")
    print(f"model_path={model_path}")
    print(probe_line)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {' '.join(str(part) for part in exc.cmd)}", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)