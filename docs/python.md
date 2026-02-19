# Correspondence to Python

## API Mapping

| Python | Julia |
|:-------|:------|
| `from structure_tensor import structure_tensor_3d` | `using StructureTensor` |
| `structure_tensor_3d(volume, sigma, rho)` | `structure_tensor_3d(volume, σ, ρ)` |
| `structure_tensor_3d(volume, sigma, rho, truncate=3.0)` | `structure_tensor_3d(volume, σ, ρ; truncate=3.0)` |
| `eig_special_3d(S)` | `eig_special_3d(S)` |
| `eig_special_3d(S, full=True)` | `eig_special_3d(S; full=true)` |
| `eig_special_3d(S, eigenvalue_order="desc")` | `eig_special_3d(S; eigenvalue_order=:desc)` |
| `from structure_tensor.cp import structure_tensor_3d` | `using CUDA; structure_tensor_3d(cu(vol), σ, ρ)` |
| `parallel_structure_tensor_analysis(data, σ, ρ, devices=['cpu'], block_size=200)` | `parallel_structure_tensor_analysis(vol, σ, ρ; chunk_size=200)` |

## Key Differences

### 1. Array Memory Layout

Julia uses **column-major** (Fortran) order; NumPy uses **row-major** (C) order.
For the same physical data, the axis ordering in memory is reversed. However,
the structure tensor computation is axis-agnostic — it computes gradients along
all three dimensions regardless of memory layout.

**In practice:** If you load the same volume in both Python and Julia, the
structure tensor and eigenvalues will be identical. The eigenvectors will
correspond to the same physical directions.

### 2. GPU Backend

| Python | Julia |
|:-------|:------|
| CuPy (`import cupy`) | CUDA.jl (`using CUDA`) |
| `cupy.ndarray` | `CuArray` |
| `cp.asnumpy(x)` | `Array(x)` |
| Auto-transfers numpy inputs | Pass `CuArray` for GPU dispatch |

### 3. Parallelism

| Python | Julia |
|:-------|:------|
| `multiprocessing` (process-based) | `Threads.@threads` (thread-based) |
| `devices` parameter (CPU/CUDA list) | Automatic (threads for CPU, CuArray for GPU) |
| Separate processes per device | Single process, multiple threads |

### 4. Structure Tensor Output

Both packages store the 6 unique components in the same order:

```
S = [Sxx, Syy, Szz, Sxy, Sxz, Syz]
```

In Python: shape `(6, x, y, z)` with C-contiguous memory.
In Julia: shape `(6, x, y, z)` with Fortran-contiguous memory.

### 5. Gradient Convention

Python labels gradient components as `Vx`, `Vy`, `Vz` corresponding to
`order=(0,0,1)`, `order=(0,1,0)`, `order=(1,0,0)` respectively. The
README notes eigenvectors are returned in "zyx" order.

Julia computes gradients along `dim 1`, `dim 2`, `dim 3` using the same
derivative order tuples. The physical meaning is identical when the same
data is provided in the same axis arrangement.

## Validation

To verify numerical equivalence between Python and Julia:

```python
# Python
import numpy as np
from structure_tensor import structure_tensor_3d, eig_special_3d

volume = np.random.RandomState(42).random((32, 32, 32))
S = structure_tensor_3d(volume, 1.5, 5.5)
val, vec = eig_special_3d(S)
np.save("py_val.npy", val)
```

```julia
# Julia
using StructureTensor, NPZ

volume = npzread("py_volume.npy")  # load same volume
S = structure_tensor_3d(volume, 1.5, 5.5)
val, vec = eig_special_3d(S)

py_val = npzread("py_val.npy")
@assert maximum(abs.(val .- py_val)) < 1e-10
```
