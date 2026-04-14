from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast


DEFAULT_CUDA_NVCC = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
DEFAULT_VSWHERE = Path(os.environ.get("ProgramFiles(x86)", r"C:/Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
DEFAULT_NINJA = Path(r"C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe")


def find_vs_installation() -> dict[str, str | None]:
    result = {
        "vswhere": str(DEFAULT_VSWHERE) if DEFAULT_VSWHERE.exists() else None,
        "installation_path": None,
        "vsdevcmd": None,
    }

    if not DEFAULT_VSWHERE.exists():
        return result

    completed = subprocess.run(
        [
            str(DEFAULT_VSWHERE),
            "-latest",
            "-products",
            "*",
            "-version",
            "[17.0,18.0)",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    installation_path = completed.stdout.strip() if completed.returncode == 0 else ""
    if installation_path:
        result["installation_path"] = installation_path
        vsdevcmd = Path(installation_path) / "Common7" / "Tools" / "VsDevCmd.bat"
        if vsdevcmd.exists():
            result["vsdevcmd"] = str(vsdevcmd)
    return result


def find_nvcc() -> dict[str, str | None]:
    candidates: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin" / "nvcc.exe")

    nvcc_on_path = shutil.which("nvcc")
    if nvcc_on_path:
        candidates.append(Path(nvcc_on_path))

    candidates.append(DEFAULT_CUDA_NVCC)

    for candidate in candidates:
        if candidate.exists():
            toolkit_root = candidate.parent.parent
            return {
                "nvcc": str(candidate.resolve()),
                "toolkit_root": str(toolkit_root.resolve()),
            }

    return {
        "nvcc": None,
        "toolkit_root": None,
    }


def find_host_compiler() -> dict[str, str | None]:
    cl_on_path = shutil.which("cl")
    return {
        "cl": cl_on_path,
    }


def find_ninja(vs_installation: dict[str, str | None]) -> dict[str, str | None]:
    ninja_on_path = shutil.which("ninja")
    if ninja_on_path:
        return {"ninja": ninja_on_path}

    installation_path = vs_installation.get("installation_path")
    if installation_path:
        candidate = Path(installation_path) / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja" / "ninja.exe"
        if candidate.exists():
            return {"ninja": str(candidate)}

    if DEFAULT_NINJA.exists():
        return {"ninja": str(DEFAULT_NINJA)}

    return {"ninja": None}


def collect_environment() -> dict[str, object]:
    vs_installation = find_vs_installation()
    nvcc = find_nvcc()
    host_compiler = find_host_compiler()
    ninja = find_ninja(vs_installation)

    return {
        "platform": sys.platform,
        "python": sys.executable,
        "cuda_path_env": os.environ.get("CUDA_PATH"),
        "vs": vs_installation,
        "cuda": nvcc,
        "host_compiler": host_compiler,
        "ninja": ninja,
        "ready": {
            "msvc": bool(vs_installation.get("vsdevcmd") or host_compiler.get("cl")),
            "cuda": bool(nvcc.get("nvcc") and nvcc.get("toolkit_root")),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the self-hosted Windows MSVC/CUDA environment used by OpenTurbo CI.")
    parser.add_argument("--require-msvc", action="store_true", help="Exit non-zero when the Visual Studio 2022 C++ toolchain is unavailable.")
    parser.add_argument("--require-cuda", action="store_true", help="Exit non-zero when nvcc or the CUDA toolkit root is unavailable.")
    parser.add_argument("--json", action="store_true", help="Print the environment report as JSON.")
    args = parser.parse_args()

    report = collect_environment()
    vs_report = cast(dict[str, Any], report["vs"])
    host_compiler_report = cast(dict[str, Any], report["host_compiler"])
    cuda_report = cast(dict[str, Any], report["cuda"])
    ninja_report = cast(dict[str, Any], report["ninja"])
    ready_report = cast(dict[str, Any], report["ready"])

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"platform={report['platform']}")
        print(f"python={report['python']}")
        print(f"vsdevcmd={vs_report['vsdevcmd']}")
        print(f"cl={host_compiler_report['cl']}")
        print(f"nvcc={cuda_report['nvcc']}")
        print(f"toolkit_root={cuda_report['toolkit_root']}")
        print(f"ninja={ninja_report['ninja']}")

    if args.require_msvc and not ready_report["msvc"]:
        return 1

    if args.require_cuda and not ready_report["cuda"]:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
