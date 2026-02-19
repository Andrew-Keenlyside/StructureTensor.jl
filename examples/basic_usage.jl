# ──────────────────────────────────────────────────────────────────────────────
# examples/basic_usage.jl — Basic structure tensor analysis examples
#
# Demonstrates the core API of StructureTensor.jl, mirroring the "Tiny
# Examples" from the Python structure-tensor README.
# ──────────────────────────────────────────────────────────────────────────────

using StructureTensor

# ═══════════════════════════════════════════════════════════════════════════════
# Example 1: Basic 3D structure tensor (equivalent to Python README)
# ═══════════════════════════════════════════════════════════════════════════════

σ = 1.5   # noise scale (inner Gaussian)
ρ = 5.5   # integration scale (outer Gaussian)

# Generate a random 3D volume (replace with your data)
volume = randn(128, 128, 128)

# Compute structure tensor
S = structure_tensor_3d(volume, σ, ρ)
# S has shape (6, 128, 128, 128)
# Components: (Sxx, Syy, Szz, Sxy, Sxz, Syz)

# Eigendecomposition (eigenvalues sorted ascending)
val, vec = eig_special_3d(S)
# val has shape (3, 128, 128, 128)  — eigenvalues (smallest first)
# vec has shape (3, 128, 128, 128)  — eigenvector for smallest eigenvalue

println("Structure tensor shape: ", size(S))
println("Eigenvalue shape: ", size(val))
println("Eigenvector shape: ", size(vec))

# ═══════════════════════════════════════════════════════════════════════════════
# Example 2: Full eigendecomposition (all three eigenvectors)
# ═══════════════════════════════════════════════════════════════════════════════

val_full, vec_full = eig_special_3d(S; full=true)
# val_full has shape (3, 128, 128, 128)
# vec_full has shape (3, 3, 128, 128, 128)
#   vec_full[:, 1, ...] = eigenvector for smallest eigenvalue
#   vec_full[:, 2, ...] = eigenvector for middle eigenvalue
#   vec_full[:, 3, ...] = eigenvector for largest eigenvalue

println("\nFull eigenvector shape: ", size(vec_full))

# ═══════════════════════════════════════════════════════════════════════════════
# Example 3: Chunked processing for large volumes
# ═══════════════════════════════════════════════════════════════════════════════

# For large volumes that don't fit in memory, use chunked processing.
# Start Julia with: julia -t 8  (for 8 threads)
println("\nUsing $(Threads.nthreads()) threads for chunked processing...")

S_chunked = structure_tensor_3d_chunked(volume, σ, ρ; chunk_size=64)
# Equivalent to structure_tensor_3d but uses less peak memory

# Combined ST + eigen in one call (most convenient for large data):
S_par, val_par, vec_par = parallel_structure_tensor_analysis(
    volume, σ, ρ;
    chunk_size = 64,
    full = false,
    verbose = true
)

println("\nDone! All examples completed successfully.")

# ═══════════════════════════════════════════════════════════════════════════════
# Example 4: Descending eigenvalue order
# ═══════════════════════════════════════════════════════════════════════════════

# The Python package sorts eigenvalues ascending by default (smallest first).
# You can also get descending order (largest first):
val_desc, vec_desc = eig_special_3d(S; eigenvalue_order=:desc)
# val_desc[1,...] is now the LARGEST eigenvalue

println("\nAscending  eigenvalues at centre: ", val[  :, 64, 64, 64])
println("Descending eigenvalues at centre: ", val_desc[:, 64, 64, 64])
