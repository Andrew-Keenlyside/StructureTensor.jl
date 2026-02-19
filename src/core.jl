# ──────────────────────────────────────────────────────────────────────────────
# core.jl — Core 3D structure tensor computation (CPU)
#
# Faithful port of `structure_tensor.st3d.structure_tensor_3d` from the Python
# `structure-tensor` package by Niels Jeppesen (Skielex).
#
# The structure tensor S of a 3D volume V at each voxel is defined as:
#
#   S = Gρ ⊛ (∇σV · ∇σVᵀ)
#
# where ∇σV is the gradient of V smoothed with a Gaussian of width σ (the
# "noise scale"), and Gρ is a Gaussian of width ρ (the "integration scale")
# that averages the outer product over a local neighbourhood.
#
# The resulting 3×3 symmetric tensor is stored as 6 unique components:
#   S[1,...] = Sxx,  S[2,...] = Syy,  S[3,...] = Szz,
#   S[4,...] = Sxy,  S[5,...] = Sxz,  S[6,...] = Syz
#
# This matches the Python package's convention:
#   S = (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz)
# ──────────────────────────────────────────────────────────────────────────────

# ── 1D Gaussian kernel (matching scipy.ndimage.gaussian_filter1d) ────────────

"""
    _gaussian_kernel1d(σ::Real, order::Int, truncate::Real) → Vector{Float64}

Compute a 1D Gaussian kernel or its first derivative, exactly matching the
behaviour of `scipy.ndimage.gaussian_filter1d`.

# Arguments
- `σ`: Standard deviation of the Gaussian (in voxels).
- `order`: Derivative order. `0` for smoothing, `1` for first derivative.
- `truncate`: Number of standard deviations at which to truncate the kernel.

# Returns
A `Vector{Float64}` containing the normalised kernel coefficients.

# Notes
For `order == 0`, returns the normalised Gaussian g(x) = exp(-x²/2σ²) / Σg.
For `order == 1`, returns g'(x) = (-x/σ²) · g(x) / Σg, where Σg is the sum
of the un-derived Gaussian. This is identical to scipy's implementation.
"""
function _gaussian_kernel1d(σ::Real, order::Int, truncate::Real)
    @assert order in (0, 1) "Only order 0 (smoothing) and 1 (first derivative) are supported."
    @assert σ > 0 "σ must be positive, got $σ."

    # Half-width of the kernel, matching scipy's `lw = int(truncate * sd + 0.5)`
    lw = round(Int, truncate * σ + 0.5)

    # Sample positions (integer offsets from centre)
    x = collect(Float64, -lw:lw)

    # Un-normalised Gaussian
    σ2 = σ * σ
    phi = @. exp(-0.5 * x^2 / σ2)

    # Normalise so that the smoothing kernel sums to 1
    phi_sum = sum(phi)
    phi ./= phi_sum

    if order == 1
        # First derivative: multiply by -x/σ² (applied to the already-normalised kernel)
        phi .*= @. -x / σ2
    end

    return phi
end

# ── Separable Gaussian derivative filtering ──────────────────────────────────

"""
    _gaussian_filter_3d!(out, volume, σ, order; truncate=4.0)

Apply a separable Gaussian (derivative) filter to a 3D volume, writing the
result into `out`. This mirrors `scipy.ndimage.gaussian_filter(volume, σ,
order=order, mode="nearest", truncate=truncate)`.

# Arguments
- `out`: Pre-allocated output array (same size as `volume`).
- `volume`: Input 3D array.
- `σ`: Gaussian standard deviation (scalar, applied isotropically).
- `order`: Tuple of three integers `(o1, o2, o3)` specifying the derivative
           order along each dimension. E.g., `(1, 0, 0)` computes ∂/∂dim1.
- `truncate`: Kernel truncation in units of σ (default `4.0`).

# Notes
Boundary handling uses `"replicate"` (nearest-neighbour extension), which
corresponds to `mode="nearest"` in scipy.
"""
function _gaussian_filter_3d!(out::AbstractArray{T,3},
                              volume::AbstractArray{<:Real,3},
                              σ::Real,
                              order::NTuple{3,Int};
                              truncate::Real = 4.0) where {T}
    # Build 1D kernels for each dimension (order specifies derivative along each axis)
    k1 = _gaussian_kernel1d(σ, order[1], truncate)
    k2 = _gaussian_kernel1d(σ, order[2], truncate)
    k3 = _gaussian_kernel1d(σ, order[3], truncate)

    # Create centred kernel factors for separable convolution.
    # ImageFiltering uses centred indexing by default with `centered()`.
    kern = (ImageFiltering.centered(k1),
            ImageFiltering.centered(k2),
            ImageFiltering.centered(k3))

    # Apply separable convolution with replicate (nearest) boundary padding.
    # This matches scipy's mode="nearest".
    imfilter!(out, volume, kern, Pad(:replicate))

    return out
end

"""
    _gaussian_filter_3d(volume, σ, order; truncate=4.0) → Array

Allocating version of [`_gaussian_filter_3d!`](@ref).
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

The structure tensor encodes the local orientation and anisotropy of
intensity gradients in a 3D image. At each voxel, the 3×3 symmetric
tensor is stored as 6 unique components.

# Arguments
- `volume::AbstractArray{<:Real, 3}`: Input 3D volume. Will be converted to
  `Float64` internally if not already floating point. Non-floating-point input
  will trigger a warning, matching the Python package's behaviour.
- `σ::Real`: Noise scale (inner scale). Standard deviation of the Gaussian
  used for computing image gradients. Controls the scale at which gradients
  are estimated — small σ captures fine detail, large σ captures coarse
  structure.
- `ρ::Real`: Integration scale (outer scale). Standard deviation of the
  Gaussian used to smooth the outer product of gradients. Controls the size
  of the neighbourhood over which orientation is averaged.

# Keyword Arguments
- `truncate::Real = 4.0`: Truncation of the Gaussian kernel in units of σ
  (or ρ). The kernel half-width is `round(Int, truncate * σ + 0.5)`. Matches
  scipy's `truncate` parameter.

# Returns
- `S::Array{Float64, 4}`: Structure tensor with shape `(6, nx, ny, nz)`.
  Components are ordered as:
  - `S[1,:,:,:]` = Sxx  (∂V/∂dim1 · ∂V/∂dim1)
  - `S[2,:,:,:]` = Syy  (∂V/∂dim2 · ∂V/∂dim2)
  - `S[3,:,:,:]` = Szz  (∂V/∂dim3 · ∂V/∂dim3)
  - `S[4,:,:,:]` = Sxy  (∂V/∂dim1 · ∂V/∂dim2)
  - `S[5,:,:,:]` = Sxz  (∂V/∂dim1 · ∂V/∂dim3)
  - `S[6,:,:,:]` = Syz  (∂V/∂dim2 · ∂V/∂dim3)

# Mathematical Background
The structure tensor at each voxel is:

    S = Gρ ⊛ (∇σV ⊗ ∇σV)

where ∇σV is the gradient computed after Gaussian smoothing with scale σ,
⊗ denotes the outer product, and Gρ is a Gaussian kernel with scale ρ
applied for spatial averaging.

# Example
```julia
using StructureTensor

volume = randn(128, 128, 128)
σ = 1.5  # noise scale
ρ = 5.5  # integration scale

S = structure_tensor_3d(volume, σ, ρ)
val, vec = eig_special_3d(S)
```

# Correspondence to Python
This function is equivalent to `structure_tensor.structure_tensor_3d(volume, sigma, rho)`.
The gradient conventions match exactly:
- `Vx` (dim 1) ↔ Python's `order=(1,0,0)` → `Vz` in Python naming
- `Vy` (dim 2) ↔ Python's `order=(0,1,0)` → `Vy` in Python naming
- `Vz` (dim 3) ↔ Python's `order=(0,0,1)` → `Vx` in Python naming

The output S is stored in the same order `(Sxx, Syy, Szz, Sxy, Sxz, Syz)`
with the axis correspondence above, yielding identical numerical results when
applied to the same data with the same axis ordering.
"""
function structure_tensor_3d(volume::AbstractArray{<:Real, 3},
                             σ::Real,
                             ρ::Real;
                             truncate::Real = 4.0)
    # ── Input validation ─────────────────────────────────────────────────
    @assert σ > 0 "σ must be positive, got $σ."
    @assert ρ > 0 "ρ must be positive, got $ρ."
    @assert ndims(volume) == 3 "Input must be a 3D array."

    # Warn if input is not floating point (matching Python behaviour)
    if !(eltype(volume) <: AbstractFloat)
        @warn "volume is not a floating-point array. This may result in " *
              "loss of precision and unexpected behaviour."
    end

    # Convert to Float64 for numerical stability
    vol = convert(Array{Float64}, volume)

    # ── Compute Gaussian-smoothed gradients ──────────────────────────────
    # Derivative along each dimension using Gaussian derivative filters.
    # This matches the Python code:
    #   Vx = gaussian_filter(volume, sigma, order=(0,0,1), ...)  ← deriv along axis 2 (last)
    #   Vy = gaussian_filter(volume, sigma, order=(0,1,0), ...)  ← deriv along axis 1
    #   Vz = gaussian_filter(volume, sigma, order=(1,0,0), ...)  ← deriv along axis 0 (first)
    #
    # In Julia (column-major), we map these to:
    #   V1 = derivative along dim 1  ↔  Python's Vz (axis 0)
    #   V2 = derivative along dim 2  ↔  Python's Vy (axis 1)
    #   V3 = derivative along dim 3  ↔  Python's Vx (axis 2)
    #
    # For identical numerical results on the SAME data layout, we use the
    # same (o1,o2,o3) tuples as the Python code. Since the user's data is
    # typically loaded with the same physical meaning per axis, the structure
    # tensor components will be consistent.

    V1 = _gaussian_filter_3d(vol, σ, (1, 0, 0); truncate = truncate)  # ∂V/∂dim1
    V2 = _gaussian_filter_3d(vol, σ, (0, 1, 0); truncate = truncate)  # ∂V/∂dim2
    V3 = _gaussian_filter_3d(vol, σ, (0, 0, 1); truncate = truncate)  # ∂V/∂dim3

    # ── Compute outer product components ─────────────────────────────────
    # Allocate output: 6 components × volume dimensions
    dims = size(vol)
    S = Array{Float64}(undef, 6, dims...)

    # Sxx = V1 * V1   (component 1)
    @views @. S[1, :, :, :] = V1 * V1
    # Syy = V2 * V2   (component 2)
    @views @. S[2, :, :, :] = V2 * V2
    # Szz = V3 * V3   (component 3)
    @views @. S[3, :, :, :] = V3 * V3
    # Sxy = V1 * V2   (component 4)
    @views @. S[4, :, :, :] = V1 * V2
    # Sxz = V1 * V3   (component 5)
    @views @. S[5, :, :, :] = V1 * V3
    # Syz = V2 * V3   (component 6)
    @views @. S[6, :, :, :] = V2 * V3

    # Free gradient arrays to reduce peak memory usage
    V1 = V2 = V3 = nothing

    # ── Smooth each component with Gρ (integration scale) ───────────────
    # Build the smoothing kernel once (no derivatives, order=(0,0,0))
    smooth_k1 = ImageFiltering.centered(_gaussian_kernel1d(ρ, 0, truncate))
    smooth_k2 = ImageFiltering.centered(_gaussian_kernel1d(ρ, 0, truncate))
    smooth_k3 = ImageFiltering.centered(_gaussian_kernel1d(ρ, 0, truncate))
    smooth_kern = (smooth_k1, smooth_k2, smooth_k3)

    # Temporary buffer for in-place smoothing
    tmp = similar(vol)
    for c in 1:6
        component = @view S[c, :, :, :]
        imfilter!(tmp, component, smooth_kern, Pad(:replicate))
        component .= tmp
    end

    return S
end
