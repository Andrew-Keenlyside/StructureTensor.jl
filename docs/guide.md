# User Guide

## Understanding the Structure Tensor

The structure tensor is a 3×3 symmetric matrix field that encodes the local orientation
and anisotropy of intensity gradients in a volumetric image. At each voxel, it is
computed as:

```math
S = G_\rho \ast (\nabla_\sigma V \otimes \nabla_\sigma V)
```

where ``\nabla_\sigma V`` is the gradient of the volume ``V`` smoothed with a Gaussian
of width ``\sigma``, ``\otimes`` is the outer product, and ``G_\rho`` averages the
result over a neighbourhood of scale ``\rho``.

## Choosing Parameters

### σ (Noise Scale / Inner Scale)

Controls the scale at which gradients are estimated:
- **Small σ (0.5–1.0):** Captures fine structures, but sensitive to noise
- **Large σ (2.0–5.0):** Captures coarse features, smooths over details
- **Recommendation:** Start with `σ = 1.0` and adjust based on your data's resolution and noise level

### ρ (Integration Scale / Outer Scale)

Controls the neighbourhood size for averaging the outer product:
- Should always be **larger than σ**
- **Small ρ:** Local orientation (less averaging)
- **Large ρ:** Regional orientation (more averaging)
- **Rule of thumb:** `ρ ≈ 2–4 × σ`

### truncate

Controls Gaussian kernel width in units of standard deviations:
- Default `4.0` is sufficient for most applications
- Larger values give more accurate filtering at the cost of computation time
- Rarely needs adjustment

## Interpreting Results

The eigenvalues ``\lambda_1 \leq \lambda_2 \leq \lambda_3`` characterise local structure:

| Eigenvalue Pattern | Interpretation |
|:---|:---|
| All ≈ 0 | Homogeneous (no gradient) |
| ``\lambda_1 \approx \lambda_2 \approx 0``, ``\lambda_3 \gg 0`` | Planar structure (edge/boundary) |
| ``\lambda_1 \approx 0``, ``\lambda_2 \approx \lambda_3 \gg 0`` | Linear structure (fibre/tube) |
| All ≈ equal, all > 0 | Isotropic (blob/noise) |

**Primary eigenvector** (``\lambda_1``): Points along the dominant linear structure
(e.g., fibre direction in tractography).

## Working with Large Volumes

For volumes exceeding available RAM, use chunked processing:

```julia
S = structure_tensor_3d_chunked(volume, σ, ρ;
    chunk_size = 200,   # voxels per block edge
    verbose = true
)
```

Or the combined pipeline:

```julia
_, val, vec = parallel_structure_tensor_analysis(
    volume, σ, ρ;
    chunk_size = 200,
    include_S = false,  # save memory
    full = true
)
```

### Block Size Guidelines

- **100–200:** Conservative memory usage, good for limited RAM
- **200–400:** Optimal speed-memory trade-off
- **If OOM:** Reduce chunk_size and/or thread count

## Multithreading

Start Julia with multiple threads:

```bash
julia -t 8
# or
export JULIA_NUM_THREADS=8
julia
```

Both chunked processing and eigendecomposition use `Threads.@threads` automatically.
