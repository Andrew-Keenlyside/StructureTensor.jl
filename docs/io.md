# File I/O

## Overview

The `process_volume` function provides an end-to-end pipeline for loading
volumetric data from neuroimaging file formats, computing the structure tensor,
and saving results.

## Supported Formats

| Format | Extension | Input | Output | Required Package |
|:-------|:----------|:-----:|:------:|:-----------------|
| TIFF (single file) | `.tif`, `.tiff` | ✅ | — | `TiffImages.jl` |
| TIFF (directory of slices) | directory | ✅ | — | `TiffImages.jl` |
| NIfTI | `.nii` | ✅ | ✅ | `NIfTI.jl` |
| NIfTI compressed | `.nii.gz` | ✅ | ✅ | `NIfTI.jl` |
| FreeSurfer MGZ | `.mgz`, `.mgh` | ✅ | ✅ | `FreeSurfer.jl` |
| Zarr | `.zarr` | ✅ | ✅ | `Zarr.jl` |
| OME-Zarr | `.ome.zarr` | ✅ | ✅ | `Zarr.jl` |
| NumPy | `.npy` | ✅ | ✅ | `NPZ.jl` |

## Installation

Install only the packages for the formats you use:

```julia
using Pkg
Pkg.add("NIfTI")        # .nii, .nii.gz
Pkg.add("TiffImages")   # .tif, .tiff
Pkg.add("FreeSurfer")   # .mgz, .mgh
Pkg.add("Zarr")         # .zarr, .ome.zarr
Pkg.add("NPZ")          # .npy
```

## Usage Examples

### NIfTI → NIfTI

```julia
using StructureTensor, NIfTI

results = process_volume("brain.nii.gz", 1.5, 5.5;
    output_dir = "results/",
    output_format = :nifti_gz,
    chunk_size = 200,
    verbose = true
)
```

### TIFF Directory → Zarr

```julia
using StructureTensor, TiffImages, Zarr

results = process_volume("path/to/slices/", 0.5, 2.0;
    output_dir = "results/",
    output_format = :zarr,
    chunk_size = 128
)
```

### Load Only

```julia
using StructureTensor, NIfTI

volume = load_volume("brain.nii.gz")
# volume is now an Array{Float64, 3}
```

### Save Only

```julia
using StructureTensor, NPZ

save_result("output.npy", val)
```

## Output Files

When `output_dir` is specified, `process_volume` creates:

- `{prefix}_tensor.{ext}` — Structure tensor (6 components)
- `{prefix}_eigenvalues.{ext}` — Eigenvalues (3 components)
- `{prefix}_eigenvectors.{ext}` — Eigenvectors (3 or 9 components)

The default prefix is `"st"`, configurable via the `prefix` keyword.
