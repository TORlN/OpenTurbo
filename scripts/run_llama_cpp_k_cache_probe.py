from __future__ import annotations

import argparse
import os
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
DEFAULT_BENCHMARK_LAYERS = "0,8,16,24,31"
DEFAULT_BENCHMARK_OUTPUT = Path("benchmarks/llama31_shadow_benchmark.md")
DEFAULT_BENCHMARK_PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "Write a haiku about debugging CUDA kernels.",
    "Summarize the benefits of grouped-query attention for large language models.",
    "Describe how to make pour-over coffee at home in four steps.",
    "List three safe warm-up stretches before Brazilian jiu-jitsu.",
    "Explain the difference between precision and recall in one paragraph.",
]

PROBE_PATTERN = re.compile(r"^\[openturbo\] cpy_k probe .*$", re.MULTILINE)
SHADOW_PATTERN = re.compile(r"^\[openturbo\] shadow_encode .*$", re.MULTILINE)
SHADOW_READ_PATTERN = re.compile(r"^\[openturbo\] shadow_read .*$", re.MULTILINE)
SHADOW_SCORE_PATTERN = re.compile(r"^\[openturbo\] shadow_score .*$", re.MULTILINE)
SHADOW_COMPARE_PATTERN = re.compile(r"^\[openturbo\] shadow_compare .*$", re.MULTILINE)
SHADOW_SNR_PATTERN = re.compile(r"^\[openturbo\] shadow_snr .*$", re.MULTILINE)
SHADOW_COMPONENTS_PATTERN = re.compile(r"^\[openturbo\] shadow_components .*$", re.MULTILINE)
SHADOW_LEGACY_PATTERN = re.compile(r"^\[openturbo\] shadow_legacy .*$", re.MULTILINE)
KEY_VALUE_PATTERN = re.compile(r"([A-Za-z_]+)=([^\s]+)")


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
    parser.add_argument(
        "--packed-score-path",
        action="store_true",
        help="Enable the experimental packed-score-path compile definition in the downstream eval-callback example.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a multi-prompt, multi-layer benchmark and write a markdown table instead of printing only one probe sample.",
    )
    parser.add_argument(
        "--benchmark-prompts-file",
        type=Path,
        help="Optional text file containing one benchmark prompt per line.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        help="Optional markdown output path for benchmark results. Defaults to benchmarks/llama31_shadow_benchmark.md under the repo root.",
    )
    parser.add_argument(
        "--layer-filter",
        default=DEFAULT_BENCHMARK_LAYERS,
        help="Comma-separated layer indices to track during benchmark runs. Default: 0,8,16,24,31.",
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


def configure_probe_build(
    cmake_exe: str,
    llama_root: Path,
    build_dir: Path,
    config: str,
    openturbo_root: Path,
    packed_score_path: bool,
    enable_cuda: bool,
) -> None:
    run_command(
        [
            cmake_exe,
            "-S",
            str(llama_root),
            "-B",
            str(build_dir),
            f"-DGGML_CUDA={'ON' if enable_cuda else 'OFF'}",
            f"-DOPENTURBO_EXPERIMENTAL_K_CACHE_PROBE=ON",
            f"-DOPENTURBO_EXPERIMENTAL_K_CACHE_SHADOW_ENCODE=ON",
            f"-DOPENTURBO_EXPERIMENTAL_PACKED_SCORE_PATH={'ON' if packed_score_path else 'OFF'}",
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


def parse_log_fields(line: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in KEY_VALUE_PATTERN.finditer(line)}


def run_probe_output(runner: Path, model_path: Path, prompt: str, seed: str, ngl: str, layer_filter: str | None = None) -> str:
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

    env = os.environ.copy()
    if layer_filter:
        env["OPENTURBO_SHADOW_LAYER_FILTER"] = layer_filter

    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=env,
    )

    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise RuntimeError(f"Probe runner failed with exit code {completed.returncode}. Output:\n{output}")
    return output


def collect_benchmark_rows(output: str, prompt_label: str) -> list[dict[str, float | int | str]]:
    compare_by_layer: dict[int, dict[str, str]] = {}
    snr_by_layer: dict[int, dict[str, str]] = {}

    for line in output.splitlines():
        if line.startswith("[openturbo] shadow_compare "):
            fields = parse_log_fields(line)
            if "layer" in fields:
                compare_by_layer[int(fields["layer"])] = fields
        elif line.startswith("[openturbo] shadow_snr "):
            fields = parse_log_fields(line)
            if "layer" in fields:
                snr_by_layer[int(fields["layer"])] = fields

    rows: list[dict[str, float | int | str]] = []
    for layer in sorted(compare_by_layer.keys() & snr_by_layer.keys()):
        compare_fields = compare_by_layer[layer]
        snr_fields = snr_by_layer[layer]
        rows.append(
            {
                "prompt": prompt_label,
                "layer": layer,
                "fwht_mae": float(compare_fields["fwht_mae"]),
                "fwht_mean_scale_ratio": float(compare_fields["fwht_mean_scale_ratio"]),
                "fwht_top_match": int(compare_fields["fwht_top_match"]),
                "signal_retention": float(snr_fields["signal_retention"]),
                "fwht_snr_db": float(snr_fields["fwht_snr_db"]),
            }
        )

    if not rows:
        raise RuntimeError("Benchmark run completed without any layer-level shadow_compare/shadow_snr rows.")

    return rows


def load_benchmark_prompts(prompts_file: Path | None) -> list[str]:
    if prompts_file is None:
        return list(DEFAULT_BENCHMARK_PROMPTS)

    prompts = [line.strip() for line in prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"Benchmark prompts file is empty: {prompts_file}")
    return prompts


def format_benchmark_report(rows: list[dict[str, float | int | str]], layer_filter: str) -> str:
    lines: list[str] = []
    lines.append("# OpenTurbo Llama Probe Benchmark")
    lines.append("")
    lines.append(f"Layers: `{layer_filter}`")
    lines.append("")
    overall_retention = sum(float(row["signal_retention"]) for row in rows) / len(rows)
    overall_snr = sum(float(row["fwht_snr_db"]) for row in rows) / len(rows)
    overall_mae = sum(float(row["fwht_mae"]) for row in rows) / len(rows)
    overall_ratio = sum(float(row["fwht_mean_scale_ratio"]) for row in rows) / len(rows)
    lines.append(f"Overall average retention: `{overall_retention:.2f}%`")
    lines.append(f"Overall average SNR: `{overall_snr:.2f} dB`")
    lines.append(f"Overall average FWHT MAE: `{overall_mae:.2f}`")
    lines.append(f"Overall average FWHT scale ratio: `{overall_ratio:.3f}`")
    lines.append("")
    lines.append("| Prompt | Layer | Retention % | SNR dB | FWHT MAE | Scale Ratio | Top Match |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['prompt']} | {int(row['layer'])} | {float(row['signal_retention']):.2f} | {float(row['fwht_snr_db']):.2f} | {float(row['fwht_mae']):.2f} | {float(row['fwht_mean_scale_ratio']):.3f} | {int(row['fwht_top_match'])} |"
        )
    return "\n".join(lines) + "\n"


def write_benchmark_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def parse_probe_output(output: str, returncode: int) -> tuple[str, str, str, str, str, str, str, str]:
    probe_match = PROBE_PATTERN.search(output)
    shadow_match = SHADOW_PATTERN.search(output)
    shadow_read_match = SHADOW_READ_PATTERN.search(output)
    shadow_score_match = SHADOW_SCORE_PATTERN.search(output)
    shadow_compare_match = SHADOW_COMPARE_PATTERN.search(output)
    shadow_snr_match = SHADOW_SNR_PATTERN.search(output)
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
    if shadow_snr_match is None:
        raise RuntimeError(
            "Probe run completed without an OpenTurbo shadow_snr line. "
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
        shadow_snr_match.group(0),
        shadow_components_match.group(0),
        shadow_legacy_match.group(0),
    )


def run_probe(runner: Path, model_path: Path, prompt: str, seed: str, ngl: str) -> tuple[str, str, str, str, str, str, str, str]:
    output = run_probe_output(runner, model_path, prompt, seed, ngl)
    return parse_probe_output(output, 0)


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
    enable_cuda = args.packed_score_path or args.ngl != "0"
    configure_probe_build(cmake_exe, llama_root, build_dir, args.config, openturbo_root, args.packed_score_path, enable_cuda)
    build_probe_target(cmake_exe, build_dir, args.config, args.target)
    runner = resolve_runner(build_dir, args.config, args.target)
    if args.model is not None:
        model_path = args.model.resolve()
    else:
        download_dir = (args.download_dir or (build_dir / "models")).resolve()
        model_path = download_hf_model(args.hf_repo, args.hf_file, download_dir)

    print(f"Running probe with model {model_path}", flush=True)

    if args.benchmark:
        prompts = load_benchmark_prompts(args.benchmark_prompts_file)
        benchmark_rows: list[dict[str, float | int | str]] = []
        try:
            base_seed = int(args.seed)
        except ValueError:
            base_seed = 42

        for prompt_index, prompt in enumerate(prompts, start=1):
            prompt_seed = str(base_seed + prompt_index - 1)
            print(f"Benchmark prompt {prompt_index}/{len(prompts)}", flush=True)
            output = run_probe_output(runner, model_path, prompt, prompt_seed, args.ngl, args.layer_filter)
            benchmark_rows.extend(collect_benchmark_rows(output, f"P{prompt_index}"))

        benchmark_report = format_benchmark_report(benchmark_rows, args.layer_filter)
        benchmark_output = (args.benchmark_output or (repo_root / DEFAULT_BENCHMARK_OUTPUT)).resolve()
        write_benchmark_report(benchmark_output, benchmark_report)
        print(f"llama_root={llama_root}")
        print(f"build_dir={build_dir}")
        print(f"model_path={model_path}")
        print(f"benchmark_output={benchmark_output}")
        print(benchmark_report)
        return 0

    probe_line, shadow_line, shadow_read_line, shadow_score_line, shadow_compare_line, shadow_snr_line, shadow_components_line, shadow_legacy_line = run_probe(runner, model_path, args.prompt, args.seed, args.ngl)

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
    print(shadow_snr_line)
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