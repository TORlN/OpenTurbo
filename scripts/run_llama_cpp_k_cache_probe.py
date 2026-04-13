from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from scaffold_llama_cpp_integration import apply_probe_patch
from scaffold_llama_cpp_integration import apply_shadow_encode_patch
from scaffold_llama_cpp_integration import ensure_llama_root_exists
from scaffold_llama_cpp_integration import resolve_output_root
from scaffold_llama_cpp_integration import resolve_llama_root
from scaffold_llama_cpp_integration import write_text_file
from scaffold_llama_cpp_integration import BRIDGE_CPP
from scaffold_llama_cpp_integration import BRIDGE_HEADER
from scaffold_llama_cpp_integration import BRIDGE_README
from scaffold_llama_cpp_integration import CMAKE_FRAGMENT
from scaffold_llama_cpp_integration import DEFAULT_LLAMA_CPP_URL
from scaffold_llama_cpp_integration import SHADOW_CALLBACK_CPP
from scaffold_llama_cpp_integration import SHADOW_CALLBACK_HEADER


DEFAULT_HF_REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
DEFAULT_HF_FILE = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
PROBE_PATTERN = re.compile(r"^\[openturbo\] cpy_k probe .*$", re.MULTILINE)
SHADOW_PATTERN = re.compile(r"^\[openturbo\] shadow_encode .*$", re.MULTILINE)
SHADOW_READ_PATTERN = re.compile(r"^\[openturbo\] shadow_read .*$", re.MULTILINE)
SHADOW_SCORE_PATTERN = re.compile(r"^\[openturbo\] shadow_score .*$", re.MULTILINE)
SHADOW_COMPARE_PATTERN = re.compile(r"^\[openturbo\] shadow_compare .*$", re.MULTILINE)
SHADOW_COMPONENTS_PATTERN = re.compile(r"^\[openturbo\] shadow_components .*$", re.MULTILINE)
SHADOW_LEGACY_PATTERN = re.compile(r"^\[openturbo\] shadow_legacy .*$", re.MULTILINE)


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
        "--model",
        type=Path,
        help="Optional local GGUF model path. If omitted, --hf-repo is used.",
    )
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help="Hugging Face repo used for downloading a GGUF locally before probe execution.",
    )
    parser.add_argument(
        "--hf-file",
        default=DEFAULT_HF_FILE,
        help="Optional Hugging Face file override passed with --hf-file.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        help="Directory for downloaded GGUF files. Defaults to <build_dir>/models.",
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


def refresh_openturbo_bindings(openturbo_root: Path) -> None:
    install_script = openturbo_root / "scripts" / "install_cuda_bindings.bat"
    if not install_script.exists():
        raise FileNotFoundError(f"Could not find OpenTurbo install script: {install_script}")

    run_command(["cmd", "/c", str(install_script)], cwd=openturbo_root)


def generate_scaffold(output_root: Path, force: bool) -> None:
    write_text_file(output_root / "openturbo_llama_cpp_bridge.hpp", BRIDGE_HEADER, force)
    write_text_file(output_root / "openturbo_llama_cpp_bridge.cpp", BRIDGE_CPP, force)
    write_text_file(output_root / "README.md", BRIDGE_README, force)
    write_text_file(output_root / "CMakeLists.fragment.txt", CMAKE_FRAGMENT, force)
    write_text_file(output_root / "openturbo_shadow_eval_callback.hpp", SHADOW_CALLBACK_HEADER, force)
    write_text_file(output_root / "openturbo_shadow_eval_callback.cpp", SHADOW_CALLBACK_CPP, force)


def configure_probe_build(cmake_exe: str, llama_root: Path, build_dir: Path, config: str, openturbo_root: Path) -> None:
    run_command(
        [
            cmake_exe,
            "-S",
            str(llama_root),
            "-B",
            str(build_dir),
            f"-DOPENTURBO_EXPERIMENTAL_K_CACHE_PROBE=ON",
            f"-DOPENTURBO_EXPERIMENTAL_K_CACHE_SHADOW_ENCODE=ON",
            f"-DOPENTURBO_ROOT={openturbo_root}",
        ]
    )


def build_probe_target(cmake_exe: str, build_dir: Path, config: str, target: str) -> None:
    run_command([cmake_exe, "--build", str(build_dir), "--config", config, "--target", target])


def download_hf_model(hf_repo: str, hf_file: str, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / Path(hf_file).name
    if destination.exists():
        return destination

    quoted_file = urllib.parse.quote(hf_file, safe="/")
    model_url = f"https://huggingface.co/{hf_repo}/resolve/main/{quoted_file}?download=true"
    print(f"Downloading {hf_repo}/{hf_file} to {destination}", flush=True)
    with urllib.request.urlopen(model_url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file, length=1024 * 1024)
    return destination


def resolve_runner(build_dir: Path, config: str, target: str) -> Path:
    exe_name = target if target.lower().endswith(".exe") else f"{target}.exe"
    return build_dir / "bin" / config / exe_name


def parse_probe_output(output: str, returncode: int) -> tuple[str, str, str, str, str, str, str]:
    probe_match = PROBE_PATTERN.search(output)
    shadow_match = SHADOW_PATTERN.search(output)
    shadow_read_match = SHADOW_READ_PATTERN.search(output)
    shadow_score_match = SHADOW_SCORE_PATTERN.search(output)
    shadow_compare_match = SHADOW_COMPARE_PATTERN.search(output)
    shadow_components_match = SHADOW_COMPONENTS_PATTERN.search(output)
    shadow_legacy_match = SHADOW_LEGACY_PATTERN.search(output)
    if probe_match is None:
        raise RuntimeError(
            "Probe run completed without an OpenTurbo cpy_k probe line. "
            f"Exit code was {returncode}."
        )
    if shadow_match is None:
        raise RuntimeError(
            "Probe run completed without an OpenTurbo shadow_encode line. "
            f"Exit code was {returncode}."
        )
    if shadow_read_match is None:
        raise RuntimeError(
            "Probe run completed without an OpenTurbo shadow_read line. "
            f"Exit code was {returncode}."
        )
    if shadow_score_match is None:
        raise RuntimeError(
            "Probe run completed without an OpenTurbo shadow_score line. "
            f"Exit code was {returncode}."
        )
    if shadow_compare_match is None:
        raise RuntimeError(
            "Probe run completed without an OpenTurbo shadow_compare line. "
            f"Exit code was {returncode}."
        )
    if shadow_components_match is None:
        raise RuntimeError(
            "Probe run completed without an OpenTurbo shadow_components line. "
            f"Exit code was {returncode}."
        )
    if shadow_legacy_match is None:
        raise RuntimeError(
            "Probe run completed without an OpenTurbo shadow_legacy line. "
            f"Exit code was {returncode}."
        )

    return (
        probe_match.group(0),
        shadow_match.group(0),
        shadow_read_match.group(0),
        shadow_score_match.group(0),
        shadow_compare_match.group(0),
        shadow_components_match.group(0),
        shadow_legacy_match.group(0),
    )


def run_probe(runner: Path, model_path: Path, prompt: str, seed: str, ngl: str) -> tuple[str, str, str, str, str, str, str]:
    command = [str(runner)]
    command.extend(["-m", str(model_path)])
    command.extend(
        [
            "--prompt",
            prompt,
            "--seed",
            seed,
            "-ngl",
            ngl,
        ]
    )

    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
    )

    output = (completed.stdout or "") + (completed.stderr or "")
    return parse_probe_output(output, completed.returncode)


def main() -> int:
    args = build_arg_parser().parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cmake_exe = resolve_cmake_executable(args.cmake)
    openturbo_root = (args.openturbo_root or repo_root).resolve()
    refresh_openturbo_bindings(openturbo_root)

    llama_root = resolve_llama_root(args)
    if llama_root is None:
        raise ValueError("A llama.cpp checkout path is required.")
    ensure_llama_root_exists(llama_root, clone_if_missing=True, clone_url=args.clone_url)
    output_root = resolve_output_root(args, llama_root)
    build_dir = (args.build_dir or (llama_root / "build-probe")).resolve()

    generate_scaffold(output_root, not args.no_force)
    apply_probe_patch(llama_root, output_root)
    apply_shadow_encode_patch(llama_root, output_root)
    configure_probe_build(cmake_exe, llama_root, build_dir, args.config, openturbo_root)
    build_probe_target(cmake_exe, build_dir, args.config, args.target)
    runner = resolve_runner(build_dir, args.config, args.target)
    if args.model is not None:
        model_path = args.model.resolve()
    else:
        download_dir = (args.download_dir or (build_dir / "models")).resolve()
        model_path = download_hf_model(args.hf_repo, args.hf_file, download_dir)

    print(f"Running probe with model {model_path}", flush=True)
    probe_line, shadow_line, shadow_read_line, shadow_score_line, shadow_compare_line, shadow_components_line, shadow_legacy_line = run_probe(runner, model_path, args.prompt, args.seed, args.ngl)

    print(f"llama_root={llama_root}")
    print(f"build_dir={build_dir}")
    print(f"model_path={model_path}")
    print(f"hf_repo={args.hf_repo}")
    print(f"hf_file={args.hf_file}")
    print(probe_line)
    print(shadow_line)
    print(shadow_read_line)
    print(shadow_score_line)
    print(shadow_compare_line)
    print(shadow_components_line)
    print(shadow_legacy_line)
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