# OpenTurbo

OpenTurbo is an experimental CUDA prototype for a TurboQuant-style 3-bit KV-cache compression pipeline with a QJL-style residual sign correction path.

The current repository is focused on kernel structure, packing layout, correctness checks, and a concrete downstream-facing head-local integration contract. It is not yet a full llama.cpp or ggml integration.

## Current State

What exists today:

* A fused encoder kernel that applies RoPE, runs a 128-wide FWHT, quantizes into a fixed 32-byte tile header, and emits a residual sign word.
* A scan kernel that estimates query-cache dot products from packed headers, including the residual correction term.
* CPU reference paths and smoke tests for the encoder and scan logic.
* A Python extension build path using CMake and scikit-build-core.
* A versioned native C ABI, a concrete head-local ggml-style adapter contract, a llama-facing request bridge, and a KV slicing shim for dense multi-head storage.

What does not exist yet:

* A production llama.cpp integration.
* A direct dependency on ggml headers or runtime types.
* End-to-end model evaluation, perplexity benchmarking, or full inference wiring.
* Portable hosted CUDA CI. Real CUDA validation is currently intended for self-hosted Windows runners.

## Architecture Snapshot

The prototype uses a fixed 128-d tile format.

Each packed tile header is 32 bytes:

* `quadrant_word_0` and `quadrant_word_1`: 64 pairs of 2-bit quadrant codes.
* `qjl_sign_word`: 64 residual sign bits.
* `block_scale_fp16`: shared block scale.
* `local_alpha_fp16`: local residual-correction scale.
* `reserved_u32`: reserved field, currently zero.

The main kernel pipeline is:

1. Apply RoPE on input pairs.
2. Run a 128-point FWHT using warp shuffles.
3. Quantize each pair into a 2-bit quadrant code.
4. Compute a shared scale and local alpha term.
5. Emit one 32-byte packed tile header.

The scan path reconstructs approximate pair centers and adds the residual sign-correlation correction:

$$
\langle q, k \rangle \approx \langle \hat{q}, \hat{k} \rangle + \alpha_k \cdot (s_q \cdot s_k)
$$

The downstream tensor contract is now frozen at a head-local slice boundary:

* Encode consumes one token's head-local values as a contiguous rank-2 view with shape `[128, num_head_tiles]`.
* Multi-tile scan consumes a contiguous cache view with shape `[num_head_tiles, num_cache_tokens]`.
* Cache layout is tile-major within each token, so all packed tiles for one token remain contiguous in memory.
* Downstream llama.cpp or ggml code is expected to slice real cache tensors into these views before calling OpenTurbo.

## Repository Layout

```text
OpenTurbo/
├── .github/workflows/        # Portable CI and optional self-hosted CUDA CI
├── .vscode/                  # Build/test tasks for smoke executables
├── include/openturbo/        # Public native headers
├── kernels/                  # CUDA kernels, C ABI, adapter layer, smoke tests
├── scripts/                  # Windows build/install scripts
├── src/openturbo/            # Python wrapper layers and pybind shim
├── tests/                    # Python tests, including CUDA smoke coverage
├── CMakeLists.txt            # Native build graph
├── pyproject.toml            # Python packaging via scikit-build-core
└── README.md
```

Important files:

* `kernels/encoder.cu`: fused encoder kernel.
* `kernels/scan.cu`: packed-header scan kernel.
* `kernels/openturbo_c_api.cu`: exported C ABI implementation.
* `kernels/openturbo_ggml_adapter.cpp`: head-local KV adapter and ranked-layout validator.
* `include/openturbo/c_api.h`: versioned public C ABI.
* `include/openturbo/ggml_adapter.h`: concrete head-local KV tensor contract for downstream ggml slicing code.
* `include/openturbo/llama_bridge.h`: dependency-free request layer matching the head-local downstream contract.
* `include/openturbo/llama_kv_shim.h`: dense multi-head KV shim that slices one head into the bridge contract.
* `src/openturbo/cuda_api.py`: raw pointer-based Python wrapper API.
* `src/openturbo/tensor_api.py`: tensor-like Python wrapper layer.

## Build And Test

### Windows CUDA Build

The Windows CUDA path is the primary native development flow in this repository.

Requirements:

* CUDA Toolkit available locally.
* Visual Studio 2022 Build Tools with 64-bit C++ tools.
* A Python environment, typically the local `.venv`.

Install the editable Python package with CUDA bindings:

```powershell
scripts\install_cuda_bindings.bat
```

Useful local smoke-test builds:

```powershell
scripts\build_smoke_test.bat
scripts\build_scan_smoke_test.bat
scripts\build_c_api_smoke_test.bat
```

Run the Python tests:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

### VS Code Tasks

The workspace includes tasks for:

* `Build CUDA Smoke Test`
* `Run CUDA Smoke Test`
* `Build CUDA Scan Smoke Test`
* `Run CUDA Scan Smoke Test`
* `Build C API Smoke Test`
* `Run C API Smoke Test`
* `Install CUDA Bindings`
* `Rebuild Bindings And Run Python Tests`

## Python Integration Layers

The Python package exposes three layers of integration:

* `openturbo.cuda_api`: raw device-pointer launch wrappers.
* `openturbo.cuda_runtime`: minimal CUDA runtime helpers for Python-side allocation, copies, and synchronization.
* `openturbo.tensor_api`: tensor-like wrappers for CUDA objects that expose `data_ptr()`.

The Python extension is `_openturbo_cuda`, built through pybind11.

## Native Integration Layers

The native side currently exposes:

* `openturbo_cuda_core`: shared kernel implementation target used by both native surfaces.
* `openturbo_c_api`: shared library exposing a stable C ABI.
* `include/openturbo/c_api.h`: public ABI with explicit version and status codes.
* `include/openturbo/ggml_adapter.h`: explicit ranked head-local adapter API.
* `include/openturbo/llama_bridge.h`: a thin request-oriented bridge layer that downstream llama.cpp code can call after slicing real cache tensors into the head-local OpenTurbo views.
* `include/openturbo/llama_kv_shim.h`: a downstream-oriented helper layer that slices dense `[... , num_heads]` KV storage into per-head or all-head bridge requests.
* `include/openturbo/ggml_downstream.hpp`: a header-only bridge that accepts real `ggml_tensor` objects once `ggml.h` is included downstream, with both single-head and all-head entry points.

The C ABI intentionally separates OpenTurbo status from raw CUDA status:

* Public functions return `openturbo_status_t`.
* `cuda_status_out` exposes the underlying CUDA error code when applicable.
* `openturbo_status_string()` and `openturbo_cuda_error_string()` provide logging-friendly text.

## Testing And CI

Local validation currently includes:

* Encoder smoke test executable.
* Scan smoke test executable.
* Native C ABI smoke test executable covering encode, bridge calls, dense multi-head KV shim calls, batched all-head dispatch, and a mock `ggml_tensor` downstream binding path.
* Native failure-path checks for malformed head-local KV layouts, invalid shim head indices, unsupported llama bridge layouts, and mismatched request counts.
* Python unit and smoke tests.

GitHub Actions currently does the following:

* Ubuntu and hosted Windows jobs run Python/package validation with `OPENTURBO_BUILD_CUDA=OFF`.
* The real Windows CUDA build is available only through manual `workflow_dispatch` on a self-hosted runner labeled `self-hosted`, `windows`, `x64`, and `cuda`.

That split is intentional: hosted runners are used for portable package validation, while actual CUDA builds are reserved for environments that already have a working CUDA toolchain and GPU path.

## Limitations

This repo should currently be treated as a kernel and integration prototype.

Known limitations:

* The repo still does not build against ggml directly; the `ggml_tensor` bridge is header-only and is meant to be compiled inside a downstream llama.cpp or ggml tree.
* The project is currently Windows-first for CUDA development.
* The Python tensor layer uses duck typing and does not enforce a specific framework.
* The packed-header format and ABI are still young enough that downstream integrations should treat them as evolving.

## Next Work

Likely next engineering steps are:

1. Wire `include/openturbo/ggml_downstream.hpp` into an actual llama.cpp integration point that owns real `ggml_tensor` KV caches.
2. Extend the shim from all-head dispatch to sequence-aware integration flows that handle multiple tokens per launch site.
3. Add model-level validation and profiler-driven performance work once the downstream bridge is exercised in a real model path.