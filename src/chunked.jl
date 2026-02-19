# ──────────────────────────────────────────────────────────────────────────────
# chunked.jl — Blocked / chunked structure tensor computation with threading
#
# Faithful port of the parallel blocked computation from the Python
# `structure-tensor` package's multiprocessing module.
#
# Large volumes are split into overlapping blocks (chunks), each processed
# independently, then stitched back together. Overlap (halo) ensures that
# Gaussian filtering at block boundaries produces correct results.
#
# Multithreading is handled via `Threads.@threads` (CPU) or via CUDA streams
# for GPU processing (see the CUDA extension).
# ──────────────────────────────────────────────────────────────────────────────

"""
    _compute_halo(σ, ρ, truncate) → Int

Compute the required halo (overlap) size for chunked processing.

The halo must be large enough to cover the support of both the inner (σ)
and outer (ρ) Gaussian kernels. We use the larger of the two kernel half-widths,
since the structure tensor computation applies both sequentially.

# Returns
The halo size in voxels (applied on each side of each chunk).
"""
function _compute_halo(σ::Real, ρ::Real, truncate::Real)
    # Kernel half-width for each scale (matching scipy's convention)
    hw_σ = round(Int, truncate * σ + 0.5)
    hw_ρ = round(Int, truncate * ρ + 0.5)
    # Total halo: both kernels are applied in sequence, so their supports add
    return hw_σ + hw_ρ
end

"""
    _chunk_ranges(total_size, chunk_size, halo) → Vector{Tuple{UnitRange, UnitRange}}

Compute the (padded_range, inner_range) pairs for splitting an axis of
length `total_size` into chunks of size `chunk_size` with `halo` overlap.

# Returns
A vector of tuples `(padded, inner)` where:
- `padded`: The range of indices to extract from the source (includes halo).
- `inner`: The range within the padded block that corresponds to the non-halo
  region (the part that will be written to the output).
"""
function _chunk_ranges(total_size::Int, chunk_size::Int, halo::Int)
    ranges = Tuple{UnitRange{Int}, UnitRange{Int}}[]

    pos = 1
    while pos <= total_size
        # Inner range (what this chunk is responsible for in the output)
        inner_start = pos
        inner_end = min(pos + chunk_size - 1, total_size)

        # Padded range (extended by halo on both sides, clamped to volume)
        pad_start = max(1, inner_start - halo)
        pad_end = min(total_size, inner_end + halo)

        # Inner range relative to the padded block
        rel_start = inner_start - pad_start + 1
        rel_end = inner_end - pad_start + 1

        push!(ranges, (pad_start:pad_end, rel_start:rel_end))

        pos = inner_end + 1
    end

    return ranges
end

"""
    structure_tensor_3d_chunked(volume, σ, ρ; chunk_size=128, truncate=4.0) → S

Compute the 3D structure tensor using blocked (chunked) processing.

This is equivalent to [`structure_tensor_3d`](@ref) but processes the volume
in overlapping blocks to reduce peak memory usage. Each block includes a halo
region to ensure Gaussian filtering at boundaries is correct.

Processing of blocks is parallelised across available Julia threads. Start
Julia with `julia -t N` or set `JULIA_NUM_THREADS=N` to use N threads.

# Arguments
- `volume`: Input 3D volume.
- `σ`: Noise scale (inner Gaussian σ).
- `ρ`: Integration scale (outer Gaussian σ).

# Keyword Arguments
- `chunk_size::Int = 128`: Size of each cubic chunk (in voxels per side).
  The actual chunk shape may differ at volume boundaries. Values between
  100 and 400 typically work well (matching Python package recommendations).
- `truncate::Real = 4.0`: Gaussian truncation parameter.
- `verbose::Bool = false`: If `true`, print progress information.

# Returns
- `S::Array{Float64, 4}`: Structure tensor with shape `(6, nx, ny, nz)`.
  Identical to the output of `structure_tensor_3d`.

# Example
```julia
# Process a large volume in 200³ blocks using 8 threads
# (start Julia with: julia -t 8)
volume = randn(512, 512, 512)
S = structure_tensor_3d_chunked(volume, 1.5, 5.5; chunk_size=200)
```

# Notes
- The halo size is computed automatically from σ, ρ, and truncate.
- Peak memory usage scales with `(chunk_size + 2*halo)³` per thread.
- For volumes that fit in memory, [`structure_tensor_3d`](@ref) may be faster
  due to lower overhead.
"""
function structure_tensor_3d_chunked(volume::AbstractArray{<:Real, 3},
                                     σ::Real,
                                     ρ::Real;
                                     chunk_size::Int = 128,
                                     truncate::Real = 4.0,
                                     verbose::Bool = false)
    @assert chunk_size > 0 "chunk_size must be positive, got $chunk_size."

    # Convert to Float64
    vol = convert(Array{Float64}, volume)
    dims = size(vol)

    # Compute required halo (overlap) for correct boundary handling
    halo = _compute_halo(σ, ρ, truncate)

    if verbose
        @info "Chunked ST: volume=$(dims), chunk_size=$chunk_size, halo=$halo, " *
              "threads=$(Threads.nthreads())"
    end

    # Compute chunk decomposition along each axis
    ranges_x = _chunk_ranges(dims[1], chunk_size, halo)
    ranges_y = _chunk_ranges(dims[2], chunk_size, halo)
    ranges_z = _chunk_ranges(dims[3], chunk_size, halo)

    # Allocate output structure tensor
    S = Array{Float64}(undef, 6, dims...)

    # Build list of all (ix, iy, iz) chunk index triples
    chunk_tasks = [(ix, iy, iz)
                   for ix in eachindex(ranges_x)
                   for iy in eachindex(ranges_y)
                   for iz in eachindex(ranges_z)]

    n_chunks = length(chunk_tasks)
    if verbose
        @info "Processing $n_chunks chunks across $(Threads.nthreads()) threads..."
    end

    # Process all chunks in parallel
    Threads.@threads for task_idx in 1:n_chunks
        ix, iy, iz = chunk_tasks[task_idx]

        # Get padded and inner ranges for this chunk
        (pad_x, inner_x) = ranges_x[ix]
        (pad_y, inner_y) = ranges_y[iy]
        (pad_z, inner_z) = ranges_z[iz]

        # Extract the padded block from the volume
        block = vol[pad_x, pad_y, pad_z]

        # Compute structure tensor on this block
        S_block = structure_tensor_3d(block, σ, ρ; truncate = truncate)

        # Copy the inner (non-halo) region to the output
        # Compute the output ranges in the full volume
        out_x = first(pad_x) + first(inner_x) - 1 : first(pad_x) + last(inner_x) - 1
        out_y = first(pad_y) + first(inner_y) - 1 : first(pad_y) + last(inner_y) - 1
        out_z = first(pad_z) + first(inner_z) - 1 : first(pad_z) + last(inner_z) - 1

        for c in 1:6
            S[c, out_x, out_y, out_z] .= @view S_block[c, inner_x, inner_y, inner_z]
        end

        if verbose && task_idx % max(1, n_chunks ÷ 10) == 0
            @info "  Progress: $task_idx / $n_chunks chunks complete"
        end
    end

    if verbose
        @info "Chunked structure tensor computation complete."
    end

    return S
end

"""
    parallel_structure_tensor_analysis(volume, σ, ρ;
        chunk_size=128, truncate=4.0, full=false,
        eigenvalue_order=:asc, verbose=false) → (S, val, vec)

Combined structure tensor computation and eigendecomposition with chunked
processing and multithreading.

This is the high-level convenience function that mirrors the Python package's
`parallel_structure_tensor_analysis`. It computes the structure tensor in
overlapping blocks and then performs the eigendecomposition, all in parallel.

# Arguments
- `volume`: Input 3D volume.
- `σ`: Noise scale (inner Gaussian σ).
- `ρ`: Integration scale (outer Gaussian σ).

# Keyword Arguments
- `chunk_size::Int = 128`: Block size for chunked processing.
- `truncate::Real = 4.0`: Gaussian truncation parameter.
- `full::Bool = false`: Return all eigenvectors (`true`) or just the primary (`false`).
- `eigenvalue_order::Symbol = :asc`: `:asc` or `:desc`.
- `include_S::Bool = true`: If `false`, do not return the structure tensor
  (saves memory for large volumes).
- `verbose::Bool = false`: Print progress information.

# Returns
- `S`: Structure tensor `(6, nx, ny, nz)` or `nothing` if `include_S=false`.
- `val`: Eigenvalues `(3, nx, ny, nz)`.
- `vec`: Eigenvectors `(3, nx, ny, nz)` or `(3, 3, nx, ny, nz)` if `full=true`.

# Example
```julia
# Analyse a large volume with 200³ blocks
S, val, vec = parallel_structure_tensor_analysis(
    volume, 1.5, 5.5;
    chunk_size = 200,
    full = true,
    verbose = true
)
```

# Correspondence to Python
Equivalent to:
```python
from structure_tensor.multiprocessing import parallel_structure_tensor_analysis
S, val, vec = parallel_structure_tensor_analysis(data, sigma, rho,
    devices=['cpu'], block_size=200)
```
"""
function parallel_structure_tensor_analysis(volume::AbstractArray{<:Real, 3},
                                            σ::Real,
                                            ρ::Real;
                                            chunk_size::Int = 128,
                                            truncate::Real = 4.0,
                                            full::Bool = false,
                                            eigenvalue_order::Symbol = :asc,
                                            include_S::Bool = true,
                                            verbose::Bool = false)
    # Step 1: Compute structure tensor (chunked)
    if verbose
        @info "Step 1/2: Computing structure tensor..."
    end
    S = structure_tensor_3d_chunked(volume, σ, ρ;
                                     chunk_size = chunk_size,
                                     truncate = truncate,
                                     verbose = verbose)

    # Step 2: Eigendecomposition (already multithreaded inside eig_special_3d)
    if verbose
        @info "Step 2/2: Computing eigendecomposition..."
    end
    val, vec = eig_special_3d(S; full = full, eigenvalue_order = eigenvalue_order)

    if verbose
        @info "Analysis complete."
    end

    if include_S
        return S, val, vec
    else
        return nothing, val, vec
    end
end
