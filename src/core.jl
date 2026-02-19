# ──────────────────────────────────────────────────────────────────────────────
# core.jl — Core 3D structure tensor computation (CPU, high-performance)
#
# Drop-in replacement for the ImageFiltering-based implementation.
# Designed to match or beat scipy.ndimage.gaussian_filter performance.
#
# Architecture:
#   • dim1 convolution: padded line-buffer (like scipy's correlate1d in C)
#     — copies each contiguous line into a padded scratch buffer, then
#       convolves with zero branches in the hot loop.  @fastmath @simd
#       vectorises across i (stride-1 reads from both buffer and output).
#
#   • dim2/dim3 convolution: scatter-accumulate pattern
#     — outer loop over kernel taps, inner @fastmath @simd loop does
#       out[i] += inp[i, jj/kk] * weight.  This is a simple axpy on
#       contiguous memory that compilers vectorise perfectly.  The clamp
#       is hoisted out of the SIMD loop entirely.
#
#   • Shared intermediates: 8 convolution passes for 3 gradients (not 9)
#   • Buffer reuse: 4 working arrays, pointer swaps instead of copies
#   • Fused product→smooth→store: one component at a time, low peak memory
#   • Multi-threaded via Threads.@threads (set JULIA_NUM_THREADS)
#
# The structure tensor S of a 3D volume V at each voxel is defined as:
#   S = Gρ ⊛ (∇σV · ∇σVᵀ)
#
# The resulting 3×3 symmetric tensor is stored as 6 unique components:
#   S[1,...] = Sxx,  S[2,...] = Syy,  S[3,...] = Szz,
#   S[4,...] = Sxy,  S[5,...] = Sxz,  S[6,...] = Syz
# ──────────────────────────────────────────────────────────────────────────────

# ── 1D Gaussian kernel (matching scipy.ndimage.gaussian_filter1d) ────────────

"""
    _gaussian_kernel1d(σ::Real, order::Int, truncate::Real) → Vector{Float64}

Compute a 1D Gaussian kernel or its first derivative, exactly matching the
behaviour of `scipy.ndimage.gaussian_filter1d`.
"""
function _gaussian_kernel1d(σ::Real, order::Int, truncate::Real)
    @assert order in (0, 1) "Only order 0 and 1 supported."
    @assert σ > 0 "σ must be positive, got $σ."

    lw = round(Int, truncate * σ + 0.5)
    x = collect(Float64, -lw:lw)
    σ2 = σ * σ
    phi = @. exp(-0.5 * x^2 / σ2)
    phi_sum = sum(phi)
    phi ./= phi_sum

    if order == 1
        phi .*= @. -x / σ2
    end

    return phi
end

# ──────────────────────────────────────────────────────────────────────────────
# CONVOLUTION ALONG DIM 1 — padded line-buffer approach
#
# This mirrors scipy's correlate1d: for each 1D line along the convolution
# axis, copy into a padded buffer with replicated boundaries, then run a
# tight dot-product loop with zero branches.
#
# The @fastmath @simd on the i-loop processes 4 doubles simultaneously
# (AVX2).  For each kernel tap m, it loads buf[i+m-1 : i+m+2] (contiguous!)
# and does an FMA with kern[m].  No clamp, no branch, no gather.
# ──────────────────────────────────────────────────────────────────────────────

function _conv1d_dim1!(out::Array{Float64,3},
                       inp::Array{Float64,3},
                       kern::Vector{Float64})
    n1, n2, n3 = size(inp)
    hk = length(kern) >> 1
    klen = length(kern)
    buflen = n1 + 2 * hk

    # Pre-allocate one padded buffer per thread
    nbufs = Threads.maxthreadid()  # safe for indexing by Threads.threadid()
    bufs = [Vector{Float64}(undef, buflen) for _ in 1:nbufs]
    

    Threads.@threads for k in 1:n3
        buf = bufs[Threads.threadid()]
        @inbounds for j in 1:n2
            # ── Fill padded buffer: [replicate_left | data | replicate_right]
            left_val  = inp[1,  j, k]
            right_val = inp[n1, j, k]
            for p in 1:hk
                buf[p] = left_val
            end
            @simd for i in 1:n1
                buf[hk + i] = inp[i, j, k]
            end
            for p in 1:hk
                buf[hk + n1 + p] = right_val
            end
            # ── Convolve: zero branches, stride-1 access on buf
            @fastmath @simd for i in 1:n1
                s = 0.0
                for m in 1:klen
                    s += buf[i + m - 1] * kern[m]
                end
                out[i, j, k] = s
            end
        end
    end
    return out
end

# ──────────────────────────────────────────────────────────────────────────────
# CONVOLUTION ALONG DIM 2 — scatter-accumulate pattern
#
# Instead of: for each output voxel, sum over kernel taps (inner loop
# prevents SIMD because of variable addressing + reduction),
#
# We do: for each kernel tap, SIMD-accumulate into the output.
#   out[:, j, k] += inp[:, clamp(j+m), k] * kern[m]
#
# The inner @simd loop is a simple  a[i] += b[i] * scalar  on contiguous
# memory — this is a textbook axpy that compilers vectorise perfectly.
# The clamp is computed once per (j, m) pair, completely outside the SIMD
# loop.
# ──────────────────────────────────────────────────────────────────────────────

function _conv1d_dim2!(out::Array{Float64,3},
                       inp::Array{Float64,3},
                       kern::Vector{Float64})
    n1, n2, n3 = size(inp)
    hk = length(kern) >> 1

    Threads.@threads for k in 1:n3
        # Zero the output slice for this k
        @inbounds for j in 1:n2
            @simd for i in 1:n1
                out[i, j, k] = 0.0
            end
        end
        # Accumulate kernel taps
        @inbounds for j in 1:n2
            for m in -hk:hk
                jj = clamp(j + m, 1, n2)
                w  = kern[m + hk + 1]
                # ── This is the hot loop: simple axpy, fully SIMD-able
                @fastmath @simd for i in 1:n1
                    out[i, j, k] += inp[i, jj, k] * w
                end
            end
        end
    end
    return out
end

# ──────────────────────────────────────────────────────────────────────────────
# CONVOLUTION ALONG DIM 3 — scatter-accumulate pattern (same idea as dim2)
#
# out[:, j, k] += inp[:, j, clamp(k+m)] * kern[m]
#
# Access inp[:, j, kk] is contiguous in i (stride 1) even though varying
# kk has stride n1*n2.  The @simd loop vectorises the i-direction.
# ──────────────────────────────────────────────────────────────────────────────

function _conv1d_dim3!(out::Array{Float64,3},
                       inp::Array{Float64,3},
                       kern::Vector{Float64})
    n1, n2, n3 = size(inp)
    hk = length(kern) >> 1

    Threads.@threads for j in 1:n2
        # Zero output for this j-plane
        @inbounds for k in 1:n3
            @simd for i in 1:n1
                out[i, j, k] = 0.0
            end
        end
        # Accumulate kernel taps
        @inbounds for k in 1:n3
            for m in -hk:hk
                kk = clamp(k + m, 1, n3)
                w  = kern[m + hk + 1]
                @fastmath @simd for i in 1:n1
                    out[i, j, k] += inp[i, j, kk] * w
                end
            end
        end
    end
    return out
end

# ── Helper: isotropic 3-pass smoothing ───────────────────────────────────────

"""
    _smooth3d!(a, b, k) → result_array

3-pass separable Gaussian smoothing.  Input in `a`.
Returns whichever buffer holds the result (always `b`).
"""
function _smooth3d!(a::Array{Float64,3}, b::Array{Float64,3},
                    k::Vector{Float64})
    _conv1d_dim1!(b, a, k)   # a → b
    _conv1d_dim2!(a, b, k)   # b → a
    _conv1d_dim3!(b, a, k)   # a → b
    return b
end

# ── Public API (backward-compatible) ─────────────────────────────────────────

"""
    _gaussian_filter_3d!(out, volume, σ, order; truncate=4.0)

Separable Gaussian (derivative) filter matching `scipy.ndimage.gaussian_filter`.
"""
function _gaussian_filter_3d!(out::AbstractArray{T,3},
                              volume::AbstractArray{<:Real,3},
                              σ::Real,
                              order::NTuple{3,Int};
                              truncate::Real = 4.0) where {T}
    k1 = _gaussian_kernel1d(σ, order[1], truncate)
    k2 = _gaussian_kernel1d(σ, order[2], truncate)
    k3 = _gaussian_kernel1d(σ, order[3], truncate)
    a = convert(Array{Float64}, volume)
    b = similar(a)
    _conv1d_dim1!(b, a, k1)
    _conv1d_dim2!(a, b, k2)
    _conv1d_dim3!(b, a, k3)
    out .= b
    return out
end

"""
    _gaussian_filter_3d(volume, σ, order; truncate=4.0) → Array
"""
function _gaussian_filter_3d(volume::AbstractArray{T,3},
                             σ::Real,
                             order::NTuple{3,Int};
                             truncate::Real = 4.0) where {T}
    out = similar(volume, Float64)
    return _gaussian_filter_3d!(out, volume, σ, order; truncate = truncate)
end

# ── Structure tensor computation ─────────────────────────────────────────────

"""
    structure_tensor_3d(volume, σ, ρ; truncate=4.0) → S

Compute the 3D structure tensor of a volume.

# Performance notes
- Set `JULIA_NUM_THREADS` for multi-threaded convolution.
- dim1 uses a padded line-buffer (like scipy's C correlate1d): zero branches
  in the hot loop, perfect SIMD vectorisation.
- dim2/dim3 use the scatter-accumulate pattern: the inner SIMD loop is a
  simple  `out[i] += inp[i] * weight`  — textbook axpy that compilers
  handle perfectly.  Boundary clamping is hoisted outside the SIMD loop.
- Gradient computation shares intermediates (8 passes instead of 9).
- Buffer reuse + pointer swaps minimise allocations.
"""
function structure_tensor_3d(volume::AbstractArray{<:Real, 3},
                             σ::Real,
                             ρ::Real;
                             truncate::Real = 4.0)
    @assert σ > 0 "σ must be positive, got $σ."
    @assert ρ > 0 "ρ must be positive, got $ρ."
    @assert ndims(volume) == 3 "Input must be a 3D array."

    if !(eltype(volume) <: AbstractFloat)
        @warn "volume is not a floating-point array. This may result in " *
              "loss of precision and unexpected behaviour."
    end

    vol = convert(Array{Float64}, volume)
    dims = size(vol)

    # ── Build kernels once ───────────────────────────────────────────────
    k_smooth = _gaussian_kernel1d(σ, 0, truncate)
    k_deriv  = _gaussian_kernel1d(σ, 1, truncate)
    k_rho    = _gaussian_kernel1d(ρ, 0, truncate)

    # ── Allocate working buffers (4 volumes, reused throughout) ──────────
    V1  = similar(vol)
    V2  = similar(vol)
    V3  = similar(vol)
    tmp = similar(vol)

    # ── Compute gradients with shared intermediates (8 passes) ───────────
    #
    #   V1 = deriv_d1(vol) → smooth_d2 → smooth_d3
    #   V2 = smooth_d1(vol) → deriv_d2 → smooth_d3
    #   V3 = smooth_d1(vol) → smooth_d2 → deriv_d3
    #
    # smooth_d1(vol) is shared between V2 and V3.

    # Shared: smooth along dim1
    _conv1d_dim1!(V2, vol, k_smooth)              #                          [1]

    # V3: smooth_d1 → smooth_d2 → deriv_d3
    _conv1d_dim2!(V3, V2, k_smooth)               #                          [2]
    _conv1d_dim3!(tmp, V3, k_deriv)               #                          [3]
    V3, tmp = tmp, V3                             # pointer swap

    # V2: smooth_d1 → deriv_d2 → smooth_d3
    _conv1d_dim2!(tmp, V2, k_deriv)               #                          [4]
    _conv1d_dim3!(V2, tmp, k_smooth)              #                          [5]

    # V1: deriv_d1 → smooth_d2 → smooth_d3
    _conv1d_dim1!(tmp, vol, k_deriv)              #                          [6]
    _conv1d_dim2!(V1, tmp, k_smooth)              #                          [7]
    _conv1d_dim3!(tmp, V1, k_smooth)              #                          [8]
    V1, tmp = tmp, V1                             # pointer swap

    # V1, V2, V3 = gradient components.  vol and tmp are free scratch.

    # ── Compute outer products, smooth with Gρ, store into S ─────────────
    S = Array{Float64}(undef, 6, dims...)
    N = prod(dims)
    S_flat   = reshape(S, 6, N)
    vol_flat = reshape(vol, N)
    V1_flat  = reshape(V1, N)
    V2_flat  = reshape(V2, N)
    V3_flat  = reshape(V3, N)

    # Process one tensor component at a time: product → smooth → store
    @inline function _process_component!(c, a_flat, b_flat)
        @inbounds @fastmath @simd for i in 1:N
            vol_flat[i] = a_flat[i] * b_flat[i]
        end
        result = _smooth3d!(vol, tmp, k_rho)
        result_flat = reshape(result, N)
        @inbounds for i in 1:N
            S_flat[c, i] = result_flat[i]
        end
    end

    _process_component!(1, V1_flat, V1_flat)    # Sxx
    _process_component!(2, V2_flat, V2_flat)    # Syy
    _process_component!(3, V3_flat, V3_flat)    # Szz
    _process_component!(4, V1_flat, V2_flat)    # Sxy
    _process_component!(5, V1_flat, V3_flat)    # Sxz
    _process_component!(6, V2_flat, V3_flat)    # Syz

    return S
end
