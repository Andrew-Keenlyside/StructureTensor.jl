# ──────────────────────────────────────────────────────────────────────────────
# examples/gpu_usage.jl — GPU-accelerated structure tensor analysis
#
# Demonstrates GPU support via CUDA.jl, mirroring the CuPy examples from
# the Python structure-tensor README.
#
# Prerequisites:
#   - NVIDIA GPU with CUDA support
#   - CUDA.jl installed: using Pkg; Pkg.add("CUDA")
# ──────────────────────────────────────────────────────────────────────────────

using StructureTensor
using CUDA

σ = 1.5
ρ = 5.5

# ═══════════════════════════════════════════════════════════════════════════════
# Example 1: GPU processing (equivalent to Python CuPy example)
# ═══════════════════════════════════════════════════════════════════════════════

# Load 3D data (on CPU)
volume = randn(128, 128, 128)

# Transfer to GPU — the structure_tensor_3d method dispatches automatically
volume_gpu = CuArray(volume)

# Compute structure tensor on GPU
S_gpu = structure_tensor_3d(volume_gpu, σ, ρ)

# Eigendecomposition on GPU
val_gpu, vec_gpu = eig_special_3d(S_gpu)

# Transfer results back to CPU
val = Array(val_gpu)
vec = Array(vec_gpu)

println("GPU structure tensor complete!")
println("  Eigenvalue shape: ", size(val))
println("  Eigenvector shape: ", size(vec))

# ═══════════════════════════════════════════════════════════════════════════════
# Example 2: Full eigenvectors on GPU
# ═══════════════════════════════════════════════════════════════════════════════

val_gpu, vec_gpu = eig_special_3d(S_gpu; full=true, eigenvalue_order=:desc)

# Move to CPU
val_full = Array(val_gpu)
vec_full = Array(vec_gpu)

println("\nFull GPU eigendecomposition:")
println("  Eigenvalue shape: ", size(val_full))
println("  Eigenvector shape: ", size(vec_full))

# ═══════════════════════════════════════════════════════════════════════════════
# Example 3: Mixed CPU/GPU workflow
# ═══════════════════════════════════════════════════════════════════════════════

# You can also pass CPU arrays directly — they're transferred automatically
S_gpu2 = structure_tensor_3d(CuArray(volume), σ, ρ)

# Free GPU memory explicitly (optional — GC handles this too)
CUDA.unsafe_free!(S_gpu)
CUDA.unsafe_free!(S_gpu2)

println("\nAll GPU examples completed successfully!")
println("GPU device: ", CUDA.name(CUDA.device()))
println("GPU memory: ", round(CUDA.available_memory() / 1e9, digits=2), " GB available")
