# OpenTurbo

OpenTurbo is an experimental CUDA prototype for a TurboQuant-style 3-bit KV-cache compression pipeline with a QJL-style residual sign correction path.

The current repository is focused on kernel structure, packing layout, correctness checks, and a concrete downstream-facing head-local integration contract. It is not yet a full llama.cpp or ggml integration.

## Current State

What exists today:

* A fused encoder kernel that applies RoPE, runs a 128-wide FWHT, quantizes into a fixed 32-byte tile header, and emits a residual sign word.
* A scan kernel that estimates query-cache dot products from packed headers, including the residual correction term.
* CPU reference paths and smoke tests for the encoder and scan logic, with the box-center scan geometry now promoted into the default estimator and the legacy corner geometry retained as an explicit comparison path.
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
* A real llama.cpp integration still requires a downstream checkout, but OpenTurbo can now generate a bridge scaffold for that tree programmatically.
* The project is currently Windows-first for CUDA development.
* The Python tensor layer uses duck typing and does not enforce a specific framework.
* The packed-header format and ABI are still young enough that downstream integrations should treat them as evolving.

## Programmatic Scaffold

If you do not want to wire the downstream bridge by hand, OpenTurbo can generate a drop-in scaffold into an existing llama.cpp checkout or a fresh destination directory:

If you do not have llama.cpp yet, the easiest one-command bootstrap on Windows is:

```powershell
scripts\bootstrap_llama_cpp_integration.bat
```

That will clone llama.cpp into a local `llama` directory if needed, generate the OpenTurbo scaffold under `llama\examples\openturbo`, and apply the experimental `cpy_k()` probe patch automatically.

If you want the full downstream probe workflow in one command, including configure, build, tiny-model download, and probe execution:

```powershell
.venv\Scripts\python.exe scripts\run_llama_cpp_k_cache_probe.py
```

That single script will:

1. Bootstrap or reuse a local llama.cpp checkout.
2. Generate the OpenTurbo scaffold.
3. Apply the experimental `cpy_k()` probe patch and the eval-callback shadow encode patch.
4. Refresh the local OpenTurbo editable install so the downstream build links against the current native library.
5. Configure and build a probe-enabled downstream tree.
6. Download the default `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` model.
7. Run `llama-eval-callback` and print the `cpy_k probe`, `shadow_encode`, `shadow_read`, `shadow_score`, and `shadow_compare` result lines.

If you run the script with no arguments, it will create a local `llama` directory in the current working directory and write the scaffold there:

```powershell
.venv\Scripts\python.exe scripts\scaffold_llama_cpp_integration.py
```

```powershell
.venv\Scripts\python.exe scripts\scaffold_llama_cpp_integration.py C:\path\to\llama.cpp
```

If you want to choose the checkout path and scaffold output path separately:

```powershell
.venv\Scripts\python.exe scripts\scaffold_llama_cpp_integration.py --llama-root C:\path\to\llama.cpp --output-dir C:\path\to\generated\openturbo
```

To clone llama.cpp and generate the scaffold in one step:

```powershell
.venv\Scripts\python.exe scripts\scaffold_llama_cpp_integration.py C:\path\to\llama.cpp --clone-if-missing
```

You can also use the Python script in explicit bootstrap mode:

```powershell
.venv\Scripts\python.exe scripts\scaffold_llama_cpp_integration.py --bootstrap
```

If you want the scaffold plus the automated downstream `cpy_k()` probe patch explicitly:

```powershell
.venv\Scripts\python.exe scripts\scaffold_llama_cpp_integration.py --bootstrap --probe-k-cache
```

That patch adds an `OPENTURBO_EXPERIMENTAL_K_CACHE_PROBE` CMake option to the downstream checkout and instruments `llama_kv_cache::cpy_k()` to emit a one-time tensor-layout log line during inference.

If you also want the execution-time shadow encode and sidecar read probe patch in the downstream eval-callback example:

```powershell
.venv\Scripts\python.exe scripts\scaffold_llama_cpp_integration.py --bootstrap --probe-k-cache --shadow-encode
```

That patch set does two additional things during downstream execution:

1. `shadow_encode`: runs the pre-rotated OpenTurbo encoder on executed `GGML_OP_SET_ROWS` K-cache writes and stores the packed headers in a per-layer sidecar keyed by cache row index.
2. `shadow_read`: observes the first executed attention node whose source chain reaches both `cache_k_l*` and `attn_inp_kq_mask`, then reports exact active-row coverage derived from the downstream mask rather than the padded K-cache view extent.
3. `shadow_score`: encodes the executed query heads in pre-rotated form, gathers the exact active sidecar rows, and runs an OpenTurbo scan estimate over those rows. On grouped-query-attention models, it now scores all query heads and maps each one onto its KV group before averaging the comparison metrics.
4. `shadow_compare`: computes a dense reference score from the executed flash-attention `q` and `k` tensors for the same exact active rows, then reports rank agreement and error metrics against `shadow_score`.
5. `shadow_components`: splits the current packed estimator into its quadrant main term and residual-correction term so remaining calibration failures can be attributed to the right part of the approximation.
6. `shadow_legacy`: logs the previous corner-based estimator as a comparison line after the box-center geometry becomes the real scan path.

At the current stage this remains a probe path, not a full attention replacement. The read-side path now uses exact mask-derived rows, and the score path is still experimental logging rather than a substitution for llama.cpp attention. The default estimator now uses the box-center geometry that previously only existed as a probe hypothesis, while `shadow_legacy` preserves the older corner reconstruction as a side-by-side diagnostic. On the current Llama-3.1-8B downstream probe, promoting box-center plus scoring all query heads in each GQA group improved the FWHT-domain mean absolute error from roughly `1.42e4` in the original corner path to about `4.21e3`, with mean scale ratio improving from about `3.33` to about `1.68`.

## Current Validation

The repo is currently validated at three levels:

* Python tests via `pytest`, including scaffold generation, downstream patch idempotency, and probe-output parsing.
* Native CUDA smoke coverage via `build\c_api_smoke_test.exe`.
* Downstream llama.cpp execution via `scripts\run_llama_cpp_k_cache_probe.py`, which now requires all seven runtime lines:
	* `cpy_k probe`
	* `shadow_encode`
	* `shadow_read`
	* `shadow_score`
	* `shadow_compare`
	* `shadow_components`
	* `shadow_legacy`

The generated files are intentionally small and explicit. They wrap real `ggml_tensor` objects through `include/openturbo/ggml_downstream.hpp`, but you still need to connect them to the actual llama.cpp call site that owns the K/V cache tensors.

## Next Work

Likely next engineering steps are:

1. Validate the promoted box-center estimator across more prompts, rows, and layers in the downstream llama probe.
2. Expand the score probe beyond the current single-stream path to broader batching coverage and richer GQA diagnostics.
3. Add model-level validation and profiler-driven performance work once the downstream bridge starts influencing attention results.