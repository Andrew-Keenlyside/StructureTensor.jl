"""
    StructureTensor

A Julia package for 3D structure tensor analysis of volumetric image data.

This is a faithful port of the Python `structure-tensor` package by Niels Jeppesen
(https://github.com/Skielex/structure-tensor), implementing identical mathematics
for computing structure tensors and their eigendecompositions on 3D volumes.

# Core API
- [`structure_tensor_3d`](@ref): Compute the 3D structure tensor of a volume.
- [`eig_special_3d`](@ref): Eigendecomposition of 3×3 symmetric structure tensors.

# Chunked Processing
- [`structure_tensor_3d_chunked`](@ref): Blocked/chunked computation for large volumes.
- [`parallel_structure_tensor_analysis`](@ref): Combined ST + eigen with multithreading.

# GPU Support
GPU-accelerated versions are available when `CUDA.jl` is loaded:
```julia
using StructureTensor
using CUDA
S = structure_tensor_3d(cu(volume), σ, ρ)
```

# I/O Wrapper
A command-line-style interface for processing neuroimaging files:
- [`process_volume`](@ref): Load → compute → save pipeline for TIFF, NIfTI, MGZ, Zarr, etc.

# References
- Jeppesen, N., et al. "Quantifying effects of manufacturing methods on fiber orientation
  in unidirectional composites using structure tensor analysis."
  *Composites Part A* 149 (2021): 106541.
"""
module StructureTensor

using LinearAlgebra
using ImageFiltering
using StaticArrays

# ── Core structure tensor computation (CPU) ──────────────────────────────────
include("core.jl")

# ── Analytical eigendecomposition via Cardano's formula ──────────────────────
include("eigen.jl")

# ── Chunked / blocked processing with multithreading ────────────────────────
include("chunked.jl")

# ── I/O wrapper for neuroimaging file formats ────────────────────────────────
include("io.jl")

# ── Public API ───────────────────────────────────────────────────────────────
export structure_tensor_3d,
       eig_special_3d,
       structure_tensor_3d_chunked,
       structure_tensor_3d_chunked_gpu,
       parallel_structure_tensor_analysis,
       process_volume

end # module StructureTensor
