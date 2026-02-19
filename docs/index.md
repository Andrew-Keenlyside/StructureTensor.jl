# StructureTensor.jl

A Julia package for fast 3D structure tensor analysis of volumetric image data.

## Overview

`StructureTensor.jl` is a faithful Julia port of the Python [`structure-tensor`](https://github.com/Skielex/structure-tensor) package by Niels Jeppesen, providing:

- **Core 3D structure tensor computation** with identical mathematics to the Python original
- **GPU acceleration** via CUDA.jl (equivalent to Python's CuPy backend)
- **Chunked/blocked processing** with multithreading for large volumes
- **File I/O wrapper** for neuroimaging formats (TIFF, NIfTI, MGZ, Zarr, OME-Zarr, NPY)

## Quick Start

```julia
using StructureTensor

volume = randn(128, 128, 128)
σ = 1.5  # noise scale
ρ = 5.5  # integration scale

S = structure_tensor_3d(volume, σ, ρ)
val, vec = eig_special_3d(S)
```

## Contents

```@contents
Pages = ["guide.md", "api.md", "gpu.md", "io.md", "python.md"]
Depth = 2
```
