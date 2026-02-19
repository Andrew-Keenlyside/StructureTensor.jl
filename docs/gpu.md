# GPU Support

## Setup

GPU acceleration requires an NVIDIA GPU with CUDA support:

```julia
using Pkg
Pkg.add("CUDA")
```

## Usage

GPU dispatch is automatic when inputs are `CuArray`:

```julia
using StructureTensor
using CUDA

volume = randn(128, 128, 128)

# Transfer to GPU
volume_gpu = CuArray(volume)

# Compute on GPU (dispatched automatically)
S = structure_tensor_3d(volume_gpu, 1.5, 5.5)
val, vec = eig_special_3d(S)

# Transfer back to CPU
val_cpu = Array(val)
vec_cpu = Array(vec)
```

## Implementation Details

The GPU implementation provides:

- **Custom CUDA kernels** for separable 1D convolution along each dimension
- **Per-voxel eigendecomposition** kernel using Cardano's formula (one thread per voxel)
- **Automatic memory management** with `CUDA.unsafe_free!` for intermediate arrays

### Convolution

Gaussian filtering uses separable 1D convolution applied sequentially along each
dimension. Each kernel uses replicate (nearest) boundary padding, matching the
CPU implementation and scipy's `mode="nearest"`.

### Eigendecomposition

The eigendecomposition kernel runs one thread per voxel, computing:
1. Cardano's formula for eigenvalues
2. Cross-product method for eigenvectors
3. In-kernel sorting (ascending or descending)

This is the same algorithm as the CPU version, adapted for GPU execution.

## Performance Tips

- GPU is most beneficial for volumes ≥ 64³ voxels
- Ensure the full volume + intermediates fit in GPU memory
- For volumes too large for GPU memory, use CPU chunked processing
- Multiple small volumes can be processed sequentially on GPU

## Memory Estimation

Approximate GPU memory required for a volume of size `N³`:

```
Memory ≈ N³ × 8 bytes × 12  (volume + 3 gradients + 6 ST components + temporaries)
```

For N=256: ~12.9 GB. For N=128: ~1.6 GB.
