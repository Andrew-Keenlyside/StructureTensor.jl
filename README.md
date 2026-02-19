> [!NOTE]
> This package is a Julia port of the Python [`structure-tensor`](https://github.com/Skielex/structure-tensor) package by Niels Jeppesen (Skielex). The mathematics and algorithmic structure are kept identical to the original.

<p align="center">
  <img src="https://raw.githubusercontent.com/Andrew-Keenlyside/StructureTensor.jl/main/docs/src/assets/banner.png" alt="StructureTensor.jl" width="700"/>
</p>

# **StructureTensor.jl**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.9%2B-purple.svg)](https://julialang.org)

A Julia package for fast 3D structure tensor analysis of volumetric image data, with CPU multithreading, GPU (CUDA) acceleration, chunked processing for large volumes, and a file I/O wrapper for neuroimaging formats.

---

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [Core Functions](#core-functions)
  - [Chunked Processing](#chunked-processing)
  - [GPU Support](#gpu-support)
  - [File I/O](#file-io)
- [User Guide](#user-guide)
  - [Understanding the Structure Tensor](#understanding-the-structure-tensor)
  - [Choosing σ and ρ](#choosing-σ-and-ρ)
  - [Working with Large Volumes](#working-with-large-volumes)
  - [GPU Acceleration](#gpu-acceleration)
  - [Multithreading](#multithreading)
  - [File Format Support](#file-format-support)
- [Correspondence to Python](#correspondence-to-python)
- [Mathematical Background](#mathematical-background)
- [Examples](#examples)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## About

The **structure tensor** is a fundamental tool in image analysis that encodes the local orientation and anisotropy of intensity gradients. It is widely used in:

- **Fibre orientation analysis** in composite materials and biological tissue
- **Tractography** and white matter mapping in neuroimaging
- **Texture analysis** and feature extraction in computer vision
- **Flow estimation** in fluid dynamics imaging

This package provides a faithful Julia port of the established Python [`structure-tensor`](https://github.com/Skielex/structure-tensor) library, with identical mathematics and several Julia-native enhancements:

| Feature | Python (`structure-tensor`) | Julia (`StructureTensor.jl`) |
|:--------|:---------------------------|:-----------------------------|
| 3D structure tensor | ✅ NumPy | ✅ Base Arrays |
| GPU acceleration | ✅ CuPy | ✅ CUDA.jl |
| Parallel/chunked | ✅ multiprocessing | ✅ Threads.@threads |
| File I/O wrapper | ❌ | ✅ TIFF, NIfTI, MGZ, Zarr, OME-Zarr, NPY |
| Eigendecomposition | ✅ Cardano's formula | ✅ Cardano's formula (identical) |

---

## Installation

### From the Julia REPL

```julia
using Pkg
Pkg.add(url="https://github.com/Andrew-Keenlyside/StructureTensor.jl")
```

### Optional Dependencies

Install these only for the features you need:

```julia
# GPU support
Pkg.add("CUDA")

# File I/O (install whichever formats you need)
Pkg.add("NIfTI")        # .nii, .nii.gz
Pkg.add("TiffImages")   # .tif, .tiff
Pkg.add("FreeSurfer")   # .mgz, .mgh
Pkg.add("Zarr")         # .zarr, .ome.zarr
Pkg.add("NPZ")          # .npy
```

### Requirements

- **Julia ≥ 1.9** (for package extensions)
- **NVIDIA GPU + CUDA toolkit** (only for GPU support)

---

## Quick Start

### Basic Usage (CPU)

```julia
using StructureTensor

σ = 1.5   # noise scale (inner Gaussian)
ρ = 5.5   # integration scale (outer Gaussian)

# Load or generate a 3D volume
volume = randn(128, 128, 128)

# Compute structure tensor
S = structure_tensor_3d(volume, σ, ρ)

# Eigendecomposition (eigenvalues sorted ascending)
val, vec = eig_special_3d(S)
```

### GPU Usage

```julia
using StructureTensor
using CUDA

volume = randn(128, 128, 128)

# Transfer to GPU and compute
S = structure_tensor_3d(CuArray(volume), 1.5, 5.5)
val, vec = eig_special_3d(S)

# Move results back to CPU
val_cpu = Array(val)
vec_cpu = Array(vec)
```

### Chunked Processing for Large Volumes

```julia
# Start Julia with threads: julia -t 8
S, val, vec = parallel_structure_tensor_analysis(
    volume, 1.5, 5.5;
    chunk_size = 200,
    verbose = true
)
```

### File I/O Pipeline

```julia
using StructureTensor
using NIfTI  # for .nii.gz support

results = process_volume(
    "brain.nii.gz", 1.5, 5.5;
    output_dir = "results/",
    output_format = :nifti_gz,
    chunk_size = 200,
    full = true,
    verbose = true
)
```

---

## API Reference

### Core Functions

#### `structure_tensor_3d(volume, σ, ρ; truncate=4.0) → S`

Compute the 3D structure tensor of a volume.

**Arguments:**
- `volume::AbstractArray{<:Real, 3}` — Input 3D volume. Converted to `Float64` internally.
- `σ::Real` — Noise scale. Standard deviation of the Gaussian for gradient estimation.
- `ρ::Real` — Integration scale. Standard deviation of the Gaussian for averaging the outer product.

**Keyword Arguments:**
- `truncate::Real = 4.0` — Gaussian kernel truncation in units of σ/ρ.

**Returns:**
- `S::Array{Float64, 4}` — Structure tensor with shape `(6, nx, ny, nz)`. Components: `(Sxx, Syy, Szz, Sxy, Sxz, Syz)`.

---

#### `eig_special_3d(S; full=false, eigenvalue_order=:asc) → (val, vec)`

Eigendecomposition of 3×3 symmetric structure tensors using Cardano's formula.

**Arguments:**
- `S::AbstractArray{<:Real, 4}` — Structure tensor `(6, nx, ny, nz)`.

**Keyword Arguments:**
- `full::Bool = false` — If `true`, return all three eigenvectors.
- `eigenvalue_order::Symbol = :asc` — `:asc` (smallest first) or `:desc` (largest first).

**Returns:**
- `val::Array{Float64, 4}` — Eigenvalues `(3, nx, ny, nz)`.
- `vec` — Eigenvectors. Shape `(3, nx, ny, nz)` if `full=false`, or `(3, 3, nx, ny, nz)` if `full=true`.

---

### Chunked Processing

#### `structure_tensor_3d_chunked(volume, σ, ρ; chunk_size=128, truncate=4.0, verbose=false) → S`

Compute the structure tensor in overlapping blocks. Reduces peak memory and enables multithreading.

**Keyword Arguments:**
- `chunk_size::Int = 128` — Block size per dimension (voxels). Values 100–400 work well.
- `verbose::Bool = false` — Print progress.

---

#### `parallel_structure_tensor_analysis(volume, σ, ρ; ...) → (S, val, vec)`

Combined structure tensor + eigendecomposition pipeline with chunked processing.

**Keyword Arguments:**
- `chunk_size::Int = 128` — Block size.
- `full::Bool = false` — All eigenvectors.
- `eigenvalue_order::Symbol = :asc` — Eigenvalue ordering.
- `include_S::Bool = true` — If `false`, returns `nothing` for S (saves memory).
- `verbose::Bool = false` — Print progress.

---

### GPU Support

When `CUDA.jl` is loaded, `structure_tensor_3d` and `eig_special_3d` dispatch automatically on `CuArray` inputs:

```julia
using StructureTensor, CUDA

S = structure_tensor_3d(cu(volume), σ, ρ)         # GPU
val, vec = eig_special_3d(S)                       # GPU
val_cpu, vec_cpu = Array(val), Array(vec)           # → CPU
```

The GPU implementation uses custom CUDA kernels with one thread per voxel for the eigendecomposition and separable 1D convolution for Gaussian filtering.

---

### File I/O

#### `process_volume(input_path, σ, ρ; ...) → NamedTuple`

End-to-end pipeline: load → compute → save.

**Keyword Arguments:**
- `output_dir` — Directory for output files (`nothing` for in-memory only).
- `output_format::Symbol` — One of `:nifti`, `:nifti_gz`, `:mgz`, `:zarr`, `:ome_zarr`, `:npy`.
- `chunk_size` — Block size for chunked processing (`nothing` for full volume).
- `full::Bool` — All eigenvectors.
- `compute_S::Bool` — Compute structure tensor.
- `compute_eigen::Bool` — Compute eigendecomposition.
- `voxel_size` — Tuple `(dx, dy, dz)` for output headers.
- `prefix::AbstractString` — Filename prefix (default `"st"`).

**Returns:** `(volume=..., S=..., val=..., vec=...)`

**Supported Formats:**

| Format | Input | Output | Required Package |
|:-------|:-----:|:------:|:-----------------|
| TIFF file (.tif/.tiff) | ✅ | — | TiffImages.jl |
| TIFF directory | ✅ | — | TiffImages.jl |
| NIfTI (.nii) | ✅ | ✅ | NIfTI.jl |
| NIfTI (.nii.gz) | ✅ | ✅ | NIfTI.jl |
| FreeSurfer MGZ (.mgz) | ✅ | ✅ | FreeSurfer.jl |
| Zarr (.zarr) | ✅ | ✅ | Zarr.jl |
| OME-Zarr (.ome.zarr) | ✅ | ✅ | Zarr.jl |
| NumPy (.npy) | ✅ | ✅ | NPZ.jl |

#### `load_volume(path) → Array{Float64, 3}`

Load a 3D volume from file. Format auto-detected from extension.

#### `save_result(path, data; header=nothing, voxel_size=nothing)`

Save an array to file. Format auto-detected from extension.

---

## User Guide

### Understanding the Structure Tensor

The structure tensor at each voxel of a 3D volume `V` is defined as:

```
S = Gρ ⊛ (∇σV · ∇σVᵀ)
```

where:
- `∇σV` is the image gradient after Gaussian smoothing with standard deviation `σ`
- `⊗` denotes the outer product
- `Gρ` is a Gaussian kernel with standard deviation `ρ` for spatial averaging

The resulting 3×3 symmetric tensor at each voxel has eigenvalues `λ₁ ≤ λ₂ ≤ λ₃` that encode local structure:

| Pattern | Meaning |
|:--------|:--------|
| `λ₁ ≈ λ₂ ≈ λ₃ ≈ 0` | Homogeneous region (no gradient) |
| `λ₁ ≈ λ₂ ≈ 0, λ₃ ≫ 0` | Planar structure (edge) |
| `λ₁ ≈ 0, λ₂ ≈ λ₃ ≫ 0` | Linear structure (fibre, tube) |
| `λ₁ ≈ λ₂ ≈ λ₃ ≫ 0` | Isotropic structure (blob, noise) |

The eigenvector corresponding to `λ₁` (smallest) points along the dominant linear structure (e.g., fibre direction).

### Choosing σ and ρ

- **σ (noise scale):** Controls the scale of gradient estimation. Small `σ` captures fine structures; large `σ` captures coarse features. Typical range: `0.5–3.0` voxels.
- **ρ (integration scale):** Controls the neighbourhood size for averaging. Should be larger than `σ`. Typical range: `2.0–10.0` voxels.
- **Rule of thumb:** `ρ ≈ 2–4 × σ` works well for most applications.

### Working with Large Volumes

For volumes that exceed available memory:

```julia
# Chunked processing uses overlapping blocks
S = structure_tensor_3d_chunked(volume, σ, ρ; chunk_size=200)

# Combined pipeline with memory-efficient option
_, val, vec = parallel_structure_tensor_analysis(
    volume, σ, ρ;
    chunk_size = 200,
    include_S = false,  # don't keep S in memory
    verbose = true
)
```

**Block size guidelines:**
- `100–200`: Conservative, lower memory usage
- `200–400`: Good balance of speed and memory
- Reduce if you encounter out-of-memory errors

### GPU Acceleration

GPU processing is beneficial for:
- Volumes larger than ~64³
- Batch processing of many volumes
- When GPU memory is sufficient (the full volume + intermediates must fit)

```julia
using CUDA

# Check available GPU memory
println(CUDA.available_memory() / 1e9, " GB available")

# For volumes that don't fit on GPU, use chunked CPU processing instead
```

### Multithreading

CPU parallel processing uses Julia's built-in threading:

```bash
# Start Julia with 8 threads
julia -t 8

# Or set the environment variable
export JULIA_NUM_THREADS=8
```

```julia
# Check thread count
println("Using $(Threads.nthreads()) threads")

# Chunked processing automatically uses all available threads
S = structure_tensor_3d_chunked(volume, σ, ρ; chunk_size=128)
```

The eigendecomposition in `eig_special_3d` is also multithreaded via `Threads.@threads`.

### File Format Support

Install only the I/O packages you need:

```julia
using Pkg

# NIfTI (most common neuroimaging format)
Pkg.add("NIfTI")

# TIFF (microscopy data)
Pkg.add("TiffImages")

# FreeSurfer MGZ
Pkg.add("FreeSurfer")

# Zarr / OME-Zarr (cloud-optimised arrays)
Pkg.add("Zarr")

# NumPy arrays
Pkg.add("NPZ")
```

---

## Correspondence to Python

This package is a faithful port of [`structure-tensor`](https://github.com/Skielex/structure-tensor). The mapping is:

| Python | Julia |
|:-------|:------|
| `from structure_tensor import structure_tensor_3d` | `using StructureTensor` |
| `structure_tensor_3d(volume, sigma, rho)` | `structure_tensor_3d(volume, σ, ρ)` |
| `eig_special_3d(S)` | `eig_special_3d(S)` |
| `eig_special_3d(S, full=True)` | `eig_special_3d(S; full=true)` |
| `from structure_tensor.cp import ...` | `using CUDA; structure_tensor_3d(cu(volume), ...)` |
| `parallel_structure_tensor_analysis(data, σ, ρ, devices=['cpu'], block_size=200)` | `parallel_structure_tensor_analysis(volume, σ, ρ; chunk_size=200)` |

**Key differences:**
1. **Array ordering:** Julia uses column-major (Fortran) order; NumPy uses row-major (C) order. The structure tensor components are stored in the same order `(Sxx, Syy, Szz, Sxy, Sxz, Syz)` with dimensions corresponding to Julia's native indexing.
2. **GPU backend:** Uses CUDA.jl (native Julia) instead of CuPy.
3. **Parallelism:** Uses Julia threads instead of Python multiprocessing. No `devices` parameter — threads handle CPU; `CuArray` dispatch handles GPU.
4. **I/O:** Additional `process_volume` wrapper not present in the Python package.

---

## Mathematical Background

### Structure Tensor

For a 3D scalar field V(x,y,z), the structure tensor is computed as:

1. **Gradient estimation:** Compute partial derivatives using Gaussian derivative filters:
   ```
   ∂V/∂xᵢ = (∂Gσ/∂xᵢ) ⊛ V
   ```
   where `Gσ` is a Gaussian with standard deviation `σ`.

2. **Outer product:** Form the 3×3 symmetric matrix at each voxel:
   ```
   T = ∇V · ∇Vᵀ
   ```

3. **Integration:** Smooth each component with a Gaussian `Gρ`:
   ```
   S = Gρ ⊛ T
   ```

### Eigendecomposition (Cardano's Formula)

For a real 3×3 symmetric matrix A with eigenvalues λ₁ ≤ λ₂ ≤ λ₃:

1. Compute `m = tr(A)/3` and shift: `K = A - mI`
2. Compute `p = ‖K‖²_F / 6` and `q = det(K) / 2`
3. Compute angle `φ = acos(clamp(q/p^{3/2}, -1, 1)) / 3`
4. Eigenvalues: `λᵢ = m + 2√p · cos(φ - 2πi/3)` for `i = 0, 1, 2`

Eigenvectors are computed via cross products of rows of `(A - λᵢI)`, selecting the pair with the largest cross-product magnitude for numerical stability.

---

## Examples

See the [`examples/`](examples/) directory:

- [`basic_usage.jl`](examples/basic_usage.jl) — Core API demonstrations
- [`gpu_usage.jl`](examples/gpu_usage.jl) — GPU acceleration with CUDA.jl
- [`io_pipeline.jl`](examples/io_pipeline.jl) — File format I/O pipeline

---

## Contributing

Contributions are welcome! Please open an [issue](https://github.com/Andrew-Keenlyside/StructureTensor.jl/issues) or [pull request](https://github.com/Andrew-Keenlyside/StructureTensor.jl/pulls).

---

## Citation

If you use this package in academic work, please cite both the original Python implementation and this Julia port:

### Original Python Package
> Jeppesen, N., et al. "Quantifying effects of manufacturing methods on fiber orientation in unidirectional composites using structure tensor analysis." *Composites Part A: Applied Science and Manufacturing* 149 (2021): 106541.

```bibtex
@article{JEPPESEN2021106541,
  title     = {Quantifying effects of manufacturing methods on fiber orientation
               in unidirectional composites using structure tensor analysis},
  journal   = {Composites Part A: Applied Science and Manufacturing},
  volume    = {149},
  pages     = {106541},
  year      = {2021},
  doi       = {https://doi.org/10.1016/j.compositesa.2021.106541},
  author    = {Jeppesen, N. and Dahl, V.A. and Christensen, A.N. and
               Dahl, A.B. and Mikkelsen, L.P.}
}
```

### This Julia Package
> Keenlyside, A. "StructureTensor.jl: Structure tensor analysis for 3D volumetric data in Julia." (2025). https://github.com/Andrew-Keenlyside/StructureTensor.jl

---

## License

[MIT License](LICENSE) — see the Python package's [LICENSE](https://github.com/Skielex/structure-tensor/blob/master/LICENSE) for the original.
