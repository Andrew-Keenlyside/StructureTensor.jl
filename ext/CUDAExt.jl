# ──────────────────────────────────────────────────────────────────────────────
# ext/CUDAExt.jl — GPU-accelerated structure tensor via CUDA.jl
#
# This package extension is loaded automatically when the user does:
#   using StructureTensor, CUDA
#
# It provides GPU-accelerated versions of `structure_tensor_3d` and
# `eig_special_3d` that operate on CuArrays. The mathematical operations
# are identical to the CPU versions — only the backend differs.
#
# Design mirrors the Python package's CuPy module (`structure_tensor.cp`),
# where CuPy functions accept both numpy and cupy arrays and automatically
# handle data transfer.
# ──────────────────────────────────────────────────────────────────────────────

module CUDAExt

using StructureTensor
using CUDA
using LinearAlgebra

# ── GPU Gaussian filtering ───────────────────────────────────────────────────

"""
    _gaussian_filter_3d_gpu!(out, volume, σ, order; truncate=4.0)

Apply a separable Gaussian (derivative) filter on the GPU.

Uses the same kernel computation as the CPU version but applies convolution
via CUDA operations. The separable 1D kernels are applied sequentially along
each dimension.
"""
function _gaussian_filter_3d_gpu!(out::CuArray{T,3},
                                  volume::CuArray{<:Real,3},
                                  σ::Real,
                                  order::NTuple{3,Int};
                                  truncate::Real = 4.0) where {T}
    # Build 1D kernels on CPU, then transfer to GPU
    k1 = StructureTensor._gaussian_kernel1d(σ, order[1], truncate)
    k2 = StructureTensor._gaussian_kernel1d(σ, order[2], truncate)
    k3 = StructureTensor._gaussian_kernel1d(σ, order[3], truncate)

    # Apply separable convolution along each dimension
    # Dimension 1
    tmp1 = _convolve_along_dim_gpu(volume, CuArray(k1), 1)
    # Dimension 2
    tmp2 = _convolve_along_dim_gpu(tmp1, CuArray(k2), 2)
    # Dimension 3
    _convolve_along_dim_gpu!(out, tmp2, CuArray(k3), 3)

    return out
end

"""
    _convolve_along_dim_gpu(volume, kernel, dim) → CuArray

Apply a 1D convolution kernel along the specified dimension of a 3D CuArray.
Uses replicate (nearest) boundary padding to match scipy's mode="nearest".
"""
function _convolve_along_dim_gpu(volume::CuArray{T,3},
                                  kernel::CuVector,
                                  dim::Int) where {T}
    out = similar(volume)
    _convolve_along_dim_gpu!(out, volume, kernel, dim)
    return out
end

function _convolve_along_dim_gpu!(out::CuArray{T,3},
                                   volume::CuArray{T,3},
                                   kernel::CuVector{Tk},
                                   dim::Int) where {T, Tk}
    klen = length(kernel)
    half = klen ÷ 2
    dims = size(volume)

    # Launch kernel with one thread per output voxel
    threads_per_block = 256
    n_elements = prod(dims)
    n_blocks = cld(n_elements, threads_per_block)

    @cuda threads=threads_per_block blocks=n_blocks _conv1d_kernel!(
        out, volume, kernel, Int32(dim), Int32(half),
        Int32(dims[1]), Int32(dims[2]), Int32(dims[3]))

    return out
end

"""
CUDA kernel for 1D convolution along a specified dimension with replicate padding.
"""
function _conv1d_kernel!(out, volume, kernel, dim, half, nx, ny, nz)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total = nx * ny * nz

    if idx <= total
        # Convert linear index to 3D coordinates (column-major)
        iz = (idx - Int32(1)) ÷ (nx * ny) + Int32(1)
        rem = (idx - Int32(1)) % (nx * ny)
        iy = rem ÷ nx + Int32(1)
        ix = rem % nx + Int32(1)

        klen = length(kernel)
        acc = zero(eltype(out))

        # Determine which dimension to convolve along
        if dim == Int32(1)
            dim_size = nx
            for k in Int32(1):klen
                # Position along the convolution dimension with replicate padding
                pos = ix + (k - half - Int32(1))
                pos = clamp(pos, Int32(1), dim_size)
                @inbounds acc += volume[pos, iy, iz] * kernel[k]
            end
        elseif dim == Int32(2)
            dim_size = ny
            for k in Int32(1):klen
                pos = iy + (k - half - Int32(1))
                pos = clamp(pos, Int32(1), dim_size)
                @inbounds acc += volume[ix, pos, iz] * kernel[k]
            end
        else  # dim == 3
            dim_size = nz
            for k in Int32(1):klen
                pos = iz + (k - half - Int32(1))
                pos = clamp(pos, Int32(1), dim_size)
                @inbounds acc += volume[ix, iy, pos] * kernel[k]
            end
        end

        @inbounds out[ix, iy, iz] = acc
    end

    return nothing
end

# ── GPU Structure Tensor ─────────────────────────────────────────────────────

"""
    StructureTensor.structure_tensor_3d(volume::CuArray, σ, ρ; truncate=4.0) → CuArray

GPU-accelerated 3D structure tensor computation.

When the input `volume` is a `CuArray`, this method is dispatched automatically.
Regular `Array` inputs are transferred to the GPU, processed, and the result
remains on the GPU as a `CuArray`. Use `Array(S)` to move results back to CPU.

# Returns
- `S::CuArray{Float64, 4}`: Structure tensor on GPU with shape `(6, nx, ny, nz)`.

# Example
```julia
using StructureTensor, CUDA

volume = CUDA.randn(128, 128, 128)
S = structure_tensor_3d(volume, 1.5, 5.5)

# Move to CPU
S_cpu = Array(S)
```
"""
function StructureTensor.structure_tensor_3d(volume::CuArray{<:Real, 3},
                                              σ::Real,
                                              ρ::Real;
                                              truncate::Real = 4.0)
    @assert σ > 0 "σ must be positive."
    @assert ρ > 0 "ρ must be positive."

    # Convert to Float64 on GPU
    vol = CuArray{Float64}(volume)
    dims = size(vol)

    # Compute Gaussian-smoothed gradients on GPU
    V1 = similar(vol)
    V2 = similar(vol)
    V3 = similar(vol)
    _gaussian_filter_3d_gpu!(V1, vol, σ, (1, 0, 0); truncate = truncate)
    _gaussian_filter_3d_gpu!(V2, vol, σ, (0, 1, 0); truncate = truncate)
    _gaussian_filter_3d_gpu!(V3, vol, σ, (0, 0, 1); truncate = truncate)

    # Compute outer product components on GPU
    S = CuArray{Float64}(undef, 6, dims...)

    # Use broadcasting on GPU views
    S_view(c) = @view S[c, :, :, :]
    @views begin
        S[1, :, :, :] .= V1 .* V1
        S[2, :, :, :] .= V2 .* V2
        S[3, :, :, :] .= V3 .* V3
        S[4, :, :, :] .= V1 .* V2
        S[5, :, :, :] .= V1 .* V3
        S[6, :, :, :] .= V2 .* V3
    end

    # Free gradient arrays
    CUDA.unsafe_free!(V1)
    CUDA.unsafe_free!(V2)
    CUDA.unsafe_free!(V3)

    # Smooth each component with ρ
    tmp = similar(vol)
    for c in 1:6
        component = CuArray{Float64}(@view S[c, :, :, :])
        _gaussian_filter_3d_gpu!(tmp, component, ρ, (0, 0, 0); truncate = truncate)
        @views S[c, :, :, :] .= tmp
        CUDA.unsafe_free!(component)
    end
    CUDA.unsafe_free!(tmp)

    return S
end

# ── GPU Eigendecomposition ───────────────────────────────────────────────────

"""
    StructureTensor.eig_special_3d(S::CuArray; full=false, eigenvalue_order=:asc)

GPU-accelerated eigendecomposition of structure tensor fields.

Uses a CUDA kernel implementing Cardano's formula, identical to the CPU version
but running on the GPU with one thread per voxel.

# Returns
- `val::CuArray{Float64}`: Eigenvalues on GPU.
- `vec::CuArray{Float64}`: Eigenvectors on GPU.
"""
function StructureTensor.eig_special_3d(S::CuArray{<:Real, 4};
                                         full::Bool = false,
                                         eigenvalue_order::Symbol = :asc)
    @assert size(S, 1) == 6 "First dimension of S must be 6."
    @assert eigenvalue_order in (:asc, :desc)

    Sf = CuArray{Float64}(S)
    dims = size(Sf)[2:end]
    nvox = prod(dims)

    val = CuArray{Float64}(undef, 3, dims...)
    if full
        vec = CuArray{Float64}(undef, 3, 3, dims...)
    else
        vec = CuArray{Float64}(undef, 3, dims...)
    end

    # Launch eigendecomposition kernel
    threads_per_block = 256
    n_blocks = cld(nvox, threads_per_block)

    is_asc = eigenvalue_order == :asc

    @cuda threads=threads_per_block blocks=n_blocks _eig_kernel!(
        val, vec, Sf, Int32(nvox), full, is_asc)

    return val, vec
end

"""
CUDA kernel for voxel-wise eigendecomposition via Cardano's formula.
"""
function _eig_kernel!(val, vec, S, nvox, full, is_asc)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if idx <= nvox
        # Extract structure tensor components
        @inbounds sxx = S[1, idx]
        @inbounds syy = S[2, idx]
        @inbounds szz = S[3, idx]
        @inbounds sxy = S[4, idx]
        @inbounds sxz = S[5, idx]
        @inbounds syz = S[6, idx]

        # Cardano's formula (identical to CPU version)
        m = (sxx + syy + szz) / 3.0

        k11 = sxx - m
        k22 = syy - m
        k33 = szz - m

        p = (k11 * k11 + k22 * k22 + k33 * k33 +
             2.0 * (sxy * sxy + sxz * sxz + syz * syz)) / 6.0

        q = (k11 * (k22 * k33 - syz * syz) -
             sxy * (sxy * k33 - syz * sxz) +
             sxz * (sxy * syz - k22 * sxz)) / 2.0

        p_sqrt = sqrt(max(p, 0.0))
        p_cubed = p * p_sqrt

        phi = if p_cubed > 0.0
            acos(clamp(q / p_cubed, -1.0, 1.0)) / 3.0
        else
            0.0
        end

        two_sqrt_p = 2.0 * p_sqrt
        e1 = m + two_sqrt_p * cos(phi)
        e2 = m + two_sqrt_p * cos(phi - 2.0 * π / 3.0)
        e3 = m + two_sqrt_p * cos(phi - 4.0 * π / 3.0)

        # Sort ascending
        λ1, λ2, λ3 = _sort3_gpu(e1, e2, e3)

        if is_asc
            @inbounds val[1, idx] = λ1
            @inbounds val[2, idx] = λ2
            @inbounds val[3, idx] = λ3
        else
            @inbounds val[1, idx] = λ3
            @inbounds val[2, idx] = λ2
            @inbounds val[3, idx] = λ1
        end

        # Eigenvectors
        if full
            if is_asc
                eigs = (λ1, λ2, λ3)
            else
                eigs = (λ3, λ2, λ1)
            end
            for vi in 1:3
                λ = if vi == 1
                    eigs[1]
                elseif vi == 2
                    eigs[2]
                else
                    eigs[3]
                end
                vx, vy, vz = _eigvec_gpu(sxx, syy, szz, sxy, sxz, syz, λ)
                @inbounds vec[1, vi, idx] = vx
                @inbounds vec[2, vi, idx] = vy
                @inbounds vec[3, vi, idx] = vz
            end
        else
            λ_small = is_asc ? λ1 : λ3
            vx, vy, vz = _eigvec_gpu(sxx, syy, szz, sxy, sxz, syz, λ_small)
            @inbounds vec[1, idx] = vx
            @inbounds vec[2, idx] = vy
            @inbounds vec[3, idx] = vz
        end
    end

    return nothing
end

# ── GPU helper functions (must be device-compatible) ─────────────────────────

@inline function _sort3_gpu(a, b, c)
    if a > b; a, b = b, a; end
    if b > c; b, c = c, b; end
    if a > b; a, b = b, a; end
    return a, b, c
end

@inline function _eigvec_gpu(sxx, syy, szz, sxy, sxz, syz, λ)
    r0x = sxx - λ;  r0y = sxy;      r0z = sxz
    r1x = sxy;      r1y = syy - λ;  r1z = syz
    r2x = sxz;      r2y = syz;      r2z = szz - λ

    c01x = r0y * r1z - r0z * r1y
    c01y = r0z * r1x - r0x * r1z
    c01z = r0x * r1y - r0y * r1x
    n01  = c01x * c01x + c01y * c01y + c01z * c01z

    c02x = r0y * r2z - r0z * r2y
    c02y = r0z * r2x - r0x * r2z
    c02z = r0x * r2y - r0y * r2x
    n02  = c02x * c02x + c02y * c02y + c02z * c02z

    c12x = r1y * r2z - r1z * r2y
    c12y = r1z * r2x - r1x * r2z
    c12z = r1x * r2y - r1y * r2x
    n12  = c12x * c12x + c12y * c12y + c12z * c12z

    if n01 >= n02 && n01 >= n12
        inv_n = 1.0 / sqrt(n01)
        return c01x * inv_n, c01y * inv_n, c01z * inv_n
    elseif n02 >= n12
        inv_n = 1.0 / sqrt(n02)
        return c02x * inv_n, c02y * inv_n, c02z * inv_n
    elseif n12 > 0.0
        inv_n = 1.0 / sqrt(n12)
        return c12x * inv_n, c12y * inv_n, c12z * inv_n
    else
        return 1.0, 0.0, 0.0
    end
end

end # module CUDAExt
