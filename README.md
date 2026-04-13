# OpenTurbo (tq3_0_qjl)

**A High-Performance CUDA Implementation of TurboQuant (Google, 2026)**

`OpenTurbo` provides an open-source, mathematically rigorous implementation of the TurboQuant KV cache compression algorithm, specifically enhanced with **QJL (Quantized Johnson-Lindenstrauss)** error correction.

This project targets the **llama.cpp / ggml** ecosystem, enabling localized inference of large language models (like Llama-3 70B) with massive context windows on consumer hardware without the catastrophic perplexity loss seen in radius-only quantization attempts.

---

## The Problem

Current 3-bit KV cache quantization methods suffer from systematic noise accumulation. In very long contexts, rounding errors develop a bias that shifts the mean of the dot product. This causes the Softmax function to hallucinate importance on irrelevant tokens.

`OpenTurbo` solves this by treating quantization as a signal processing problem, utilizing an orthogonal rotation to smear outliers and a 1-bit residual correction to preserve the inner-product.

---

## Specifications

### 1. Random Orthogonal Rotation (FWHT)
To ensure the 2nd-bit angle quantization is efficient, we must first "smear" the energy of the vector. We implement a **Fast Walsh-Hadamard Transform (FWHT)** using CUDA Warp Shuffles (`__shfl_xor_sync`).
* **Complexity:** $O(d \log d)$
* **Hardware:** Zero floating-point multiplications; utilizes addition/subtraction butterflies.
* **RoPE Interaction:** RoPE is applied before the FWHT to ensure positional geometry is preserved in the transformed space.

### 2. 2-bit Polar Quantization
Once rotated, the vector elements follow a Gaussian distribution. We map pairs of dimensions $(x, y)$ to:
* **Radius ($R$):** A shared high-precision scale (FP16/E4M3) per block.
* **Angle ($\theta$):** A 2-bit value representing one of four quadrants ($45^\circ, 135^\circ, 225^\circ, 315^\circ$).

### 3. QJL 1-bit Sign Correction
This is the "1-bit nudge." We calculate the residual error $e = u_{original} - \hat{u}_{quantized}$ and store the sign of the aggregate error.
* **Correction Formula:** $\langle u, v \rangle \approx \langle \hat{u}, \hat{v} \rangle + \alpha (s_u \cdot s_v)$
* This ensures that the "direction" of the rounding error is recovered during the attention scan, preventing the Mean Shift.

---

## Project Structure

```text
OPENTURBO/
├── .github/workflows/   # CI/CD for CUDA/Python builds
├── src/openturbo/       # Python bindings and CLI
│   └── *.py             # Model conversion tools
├── tests/               # Validation vs. FP16 baselines
├── kernels/             # (TBP) Custom CUDA source (.cu / .cuh)
├── pyproject.toml       # Build system requirements
└── README.md            # You are here
```

## Building The Python CUDA Extension

The project now includes a native-extension build path for the `_openturbo_cuda`
module using CMake + scikit-build-core.

```powershell
scripts\install_cuda_bindings.bat
```

On this workspace, the build assumes:

* CUDA Toolkit is installed and available at the standard Windows location.
* Visual Studio 2022 Build Tools are available for 64-bit host builds.
* The target architecture is RTX 4090 (`sm_89`) unless overridden with `CMAKE_CUDA_ARCHITECTURES`.

## Python Integration Layers

The package now exposes three progressively higher-level entry points:

* Raw launch wrappers in `openturbo.cuda_api` for integer device pointers.
* Minimal CUDA runtime helpers in `openturbo.cuda_runtime` for Python-side device allocation and copies.
* Tensor-style bridge helpers in `openturbo.tensor_api` for CUDA tensor-like objects that expose `data_ptr()`.

## Native Integration Layers

The CUDA build now exposes two native integration surfaces above the kernel wrappers:

* A shared internal CUDA core used by both the Python extension and the exported C ABI.
* An exported C header at `include/openturbo/c_api.h` for future ggml / llama.cpp integration without pybind11.