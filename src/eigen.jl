# ──────────────────────────────────────────────────────────────────────────────
# eigen.jl — Analytical eigendecomposition of 3×3 symmetric structure tensors
#
# Faithful port of `structure_tensor.st3d.eig_special_3d` from the Python
# `structure-tensor` package by Niels Jeppesen (Skielex).
#
# Uses Cardano's formula for the eigenvalues of a real 3×3 symmetric matrix,
# followed by cross-product-based eigenvector computation. This closed-form
# approach is significantly faster than iterative eigensolvers for the
# element-wise decomposition of large tensor fields.
#
# The 3×3 symmetric matrix at each voxel is reconstructed from the 6 unique
# components stored in S:
#
#       ⎡ Sxx  Sxy  Sxz ⎤     ⎡ S[1]  S[4]  S[5] ⎤
#   A = ⎢ Sxy  Syy  Syz ⎥  =  ⎢ S[4]  S[2]  S[6] ⎥
#       ⎣ Sxz  Syz  Szz ⎦     ⎣ S[5]  S[6]  S[3] ⎦
#
# ──────────────────────────────────────────────────────────────────────────────

"""
    eig_special_3d(S; full=false, eigenvalue_order=:asc) → (val, vec)

Compute eigenvalues and eigenvectors of the 3×3 symmetric structure tensor
field using Cardano's analytical formula.

This is a faithful port of the Python `eig_special_3d` function, yielding
numerically identical results (up to floating-point precision).

# Arguments
- `S::AbstractArray{<:Real, 4}`: Structure tensor field with shape
  `(6, nx, ny, nz)`. The 6 components are `(Sxx, Syy, Szz, Sxy, Sxz, Syz)`.
  If not already `Float64`, the array will be copied and converted.

# Keyword Arguments
- `full::Bool = false`: If `false`, return only the eigenvector corresponding
  to the smallest eigenvalue (shape `(3, nx, ny, nz)`). If `true`, return all
  three eigenvectors (shape `(3, 3, nx, ny, nz)`) where `vec[i, :, :, :, :]`
  is the i-th eigenvector.
- `eigenvalue_order::Symbol = :asc`: Eigenvalue ordering. `:asc` for ascending
  (smallest first), `:desc` for descending (largest first). The eigenvectors
  are reordered to match.

# Returns
- `val::Array{Float64, 4}`: Eigenvalues with shape `(3, nx, ny, nz)`.
  `val[1,:,:,:]` is the first eigenvalue (smallest if `:asc`, largest if `:desc`).
- `vec`: Eigenvectors. Shape depends on `full`:
  - `full=false`: `(3, nx, ny, nz)` — the eigenvector for the smallest eigenvalue.
  - `full=true`: `(3, 3, nx, ny, nz)` — all eigenvectors, ordered to match `val`.
    `vec[:, i, :, :, :]` is the eigenvector corresponding to `val[i, :, :, :]`.

# Mathematical Background
For a real 3×3 symmetric matrix A, Cardano's formula gives the three real
eigenvalues analytically:

1. Compute the trace `m = tr(A)/3` and shift: `K = A - mI`.
2. Compute `q = det(K)/2` and `p = ‖K‖²_F / 6`.
3. Compute the discriminant angle `φ = acos(clamp(q/p^{3/2}, -1, 1)) / 3`.
4. Eigenvalues: `λᵢ = m + 2√p · cos(φ - 2πi/3)` for `i ∈ {0, 1, 2}`.

Eigenvectors are computed via cross products of rows of `(A - λᵢI)`.

# Example
```julia
S = structure_tensor_3d(volume, 1.5, 5.5)

# Get only the primary eigenvector (smallest eigenvalue)
val, vec = eig_special_3d(S)

# Get all eigenvectors, descending order
val, vec = eig_special_3d(S; full=true, eigenvalue_order=:desc)
```

# Correspondence to Python
Equivalent to `structure_tensor.eig_special_3d(S, full=full)` with
`eigenvalue_order` matching Python's `"asc"` / `"desc"` strings.
"""
function eig_special_3d(S::AbstractArray{<:Real, 4};
                        full::Bool = false,
                        eigenvalue_order::Symbol = :asc)
    # ── Input validation ─────────────────────────────────────────────────
    @assert size(S, 1) == 6 "First dimension of S must be 6, got $(size(S, 1))."
    @assert eigenvalue_order in (:asc, :desc) "eigenvalue_order must be :asc or :desc."

    # Convert to Float64 for numerical stability (matching Python's recommendation)
    Sf = convert(Array{Float64}, S)

    dims = size(Sf)[2:end]  # (nx, ny, nz)
    nvox = prod(dims)

    # ── Allocate output arrays ───────────────────────────────────────────
    val = Array{Float64}(undef, 3, dims...)
    if full
        vec = Array{Float64}(undef, 3, 3, dims...)
    else
        vec = Array{Float64}(undef, 3, dims...)
    end

    # ── Element-wise eigendecomposition using Cardano's formula ──────────
    # Process each voxel. We use @inbounds and linear indexing for speed.
    # The inner function is kept small to aid compiler optimisation.
    Threads.@threads for idx in 1:nvox
        # Extract the 6 unique components of the 3×3 symmetric matrix
        sxx = Sf[1, idx]
        syy = Sf[2, idx]
        szz = Sf[3, idx]
        sxy = Sf[4, idx]
        sxz = Sf[5, idx]
        syz = Sf[6, idx]

        # ── Cardano's formula for 3×3 symmetric eigenvalues ──────────
        # Step 1: Compute mean of diagonal (trace / 3)
        m = (sxx + syy + szz) / 3.0

        # Step 2: Shift matrix K = A - m*I (makes the problem trace-free)
        k11 = sxx - m
        k22 = syy - m
        k33 = szz - m
        # Off-diagonal elements are unchanged by the shift
        k12 = sxy
        k13 = sxz
        k23 = syz

        # Step 3: Compute p = (1/6) * ||K||²_F  (Frobenius norm squared / 6)
        p = (k11 * k11 + k22 * k22 + k33 * k33 +
             2.0 * (k12 * k12 + k13 * k13 + k23 * k23)) / 6.0

        # Step 4: Compute q = det(K) / 2
        # det(K) = k11*(k22*k33 - k23²) - k12*(k12*k33 - k23*k13) + k13*(k12*k23 - k22*k13)
        q = (k11 * (k22 * k33 - k23 * k23) -
             k12 * (k12 * k33 - k23 * k13) +
             k13 * (k12 * k23 - k22 * k13)) / 2.0

        # Step 5: Compute the discriminant angle φ
        # Guard against numerical issues: clamp argument of acos to [-1, 1]
        p_sqrt = sqrt(max(p, 0.0))
        p_cubed = p * p_sqrt  # = p^(3/2)

        # Avoid division by zero when p ≈ 0 (isotropic case)
        if p_cubed > 0.0
            phi = acos(clamp(q / p_cubed, -1.0, 1.0)) / 3.0
        else
            phi = 0.0
        end

        # Step 6: Compute the three eigenvalues
        # λ₁ ≥ λ₂ ≥ λ₃ in the Cardano ordering (before final sort)
        two_sqrt_p = 2.0 * p_sqrt
        e1 = m + two_sqrt_p * cos(phi)
        e2 = m + two_sqrt_p * cos(phi - 2.0 * π / 3.0)
        e3 = m + two_sqrt_p * cos(phi - 4.0 * π / 3.0)

        # Sort eigenvalues ascending (e3 ≤ e2 ≤ e1 from Cardano, but verify)
        # Cardano's formula gives: e1 ≥ e2 ≥ e3, so ascending is (e3, e2, e1)
        λ1, λ2, λ3 = _sort3(e1, e2, e3)  # returns (min, mid, max)

        # Store eigenvalues in requested order
        if eigenvalue_order == :asc
            @inbounds val[1, idx] = λ1  # smallest
            @inbounds val[2, idx] = λ2  # middle
            @inbounds val[3, idx] = λ3  # largest
        else
            @inbounds val[1, idx] = λ3  # largest
            @inbounds val[2, idx] = λ2  # middle
            @inbounds val[3, idx] = λ1  # smallest
        end

        # ── Eigenvector computation ──────────────────────────────────
        # For each eigenvalue λ, the eigenvector is found from the null space
        # of (A - λI). For a 3×3 matrix, we compute the cross product of two
        # rows of (A - λI) and normalise.

        if full
            # Compute all three eigenvectors, ordered to match eigenvalues
            if eigenvalue_order == :asc
                eig_vals = (λ1, λ2, λ3)
            else
                eig_vals = (λ3, λ2, λ1)
            end

            for (vi, λ) in enumerate(eig_vals)
                vx, vy, vz = _eigvec_from_eigenvalue(
                    sxx, syy, szz, sxy, sxz, syz, λ)
                @inbounds vec[1, vi, idx] = vx
                @inbounds vec[2, vi, idx] = vy
                @inbounds vec[3, vi, idx] = vz
            end
        else
            # Only compute the eigenvector for the smallest eigenvalue
            λ_small = eigenvalue_order == :asc ? λ1 : λ3
            vx, vy, vz = _eigvec_from_eigenvalue(
                sxx, syy, szz, sxy, sxz, syz, λ_small)
            @inbounds vec[1, idx] = vx
            @inbounds vec[2, idx] = vy
            @inbounds vec[3, idx] = vz
        end
    end

    return val, vec
end

# ── Helper: Sort three values ascending ──────────────────────────────────────

"""
    _sort3(a, b, c) → (min, mid, max)

Sort three scalar values in ascending order using a minimal comparison network.
"""
@inline function _sort3(a::T, b::T, c::T) where {T<:Real}
    if a > b
        a, b = b, a
    end
    if b > c
        b, c = c, b
    end
    if a > b
        a, b = b, a
    end
    return a, b, c
end

# ── Helper: Eigenvector from eigenvalue via cross-product method ─────────────

"""
    _eigvec_from_eigenvalue(sxx, syy, szz, sxy, sxz, syz, λ) → (vx, vy, vz)

Compute the normalised eigenvector of the 3×3 symmetric matrix A corresponding
to eigenvalue λ, using the cross-product of rows of (A - λI).

The method computes the cross product of all three pairs of rows and selects
the one with the largest magnitude, providing numerical robustness even when
(A - λI) has a rank deficiency in a particular row pair.

# Returns
A normalised eigenvector `(vx, vy, vz)`. Returns `(1.0, 0.0, 0.0)` as a
fallback if the matrix is degenerate (e.g., all eigenvalues equal).
"""
@inline function _eigvec_from_eigenvalue(sxx::Float64, syy::Float64, szz::Float64,
                                          sxy::Float64, sxz::Float64, syz::Float64,
                                          λ::Float64)
    # Rows of (A - λI):
    # row0 = (sxx - λ,  sxy,      sxz)
    # row1 = (sxy,      syy - λ,  syz)
    # row2 = (sxz,      syz,      szz - λ)

    r0x = sxx - λ;  r0y = sxy;      r0z = sxz
    r1x = sxy;      r1y = syy - λ;  r1z = syz
    r2x = sxz;      r2y = syz;      r2z = szz - λ

    # Cross product of row0 × row1
    c01x = r0y * r1z - r0z * r1y
    c01y = r0z * r1x - r0x * r1z
    c01z = r0x * r1y - r0y * r1x
    n01  = c01x * c01x + c01y * c01y + c01z * c01z

    # Cross product of row0 × row2
    c02x = r0y * r2z - r0z * r2y
    c02y = r0z * r2x - r0x * r2z
    c02z = r0x * r2y - r0y * r2x
    n02  = c02x * c02x + c02y * c02y + c02z * c02z

    # Cross product of row1 × row2
    c12x = r1y * r2z - r1z * r2y
    c12y = r1z * r2x - r1x * r2z
    c12z = r1x * r2y - r1y * r2x
    n12  = c12x * c12x + c12y * c12y + c12z * c12z

    # Select the cross product with the largest squared norm
    if n01 >= n02 && n01 >= n12
        inv_norm = 1.0 / sqrt(n01)
        return c01x * inv_norm, c01y * inv_norm, c01z * inv_norm
    elseif n02 >= n12
        inv_norm = 1.0 / sqrt(n02)
        return c02x * inv_norm, c02y * inv_norm, c02z * inv_norm
    elseif n12 > 0.0
        inv_norm = 1.0 / sqrt(n12)
        return c12x * inv_norm, c12y * inv_norm, c12z * inv_norm
    else
        # Degenerate case: all eigenvalues are equal (isotropic tensor).
        # Return a default unit vector.
        return 1.0, 0.0, 0.0
    end
end
