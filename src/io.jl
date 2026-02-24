# ──────────────────────────────────────────────────────────────────────────────
# io.jl — I/O wrapper for loading and saving volumetric data
#
# Provides a high-level `process_volume` function that accepts various
# neuroimaging file formats as input, computes the structure tensor and
# eigendecomposition, and saves results in the desired output format.
#
# Supported input formats:
#   - Single TIFF file (.tif, .tiff)
#   - Directory of TIFF slices
#   - NIfTI (.nii, .nii.gz)
#   - FreeSurfer MGZ (.mgz)
#   - Zarr (.zarr)
#   - OME-Zarr (.ome.zarr)
#
# Supported output formats:
#   - NIfTI (.nii, .nii.gz)
#   - FreeSurfer MGZ (.mgz)
#   - Zarr (.zarr)
#   - OME-Zarr (.ome.zarr)
#   - NumPy (.npy)
#   - Raw Julia array (returned in memory)
#
# Dependencies are loaded lazily via `@require` / package extensions to keep
# the core package lightweight. Users must install the relevant I/O packages
# themselves:
#   - TiffImages.jl     → for TIFF
#   - NIfTI.jl          → for NIfTI (.nii, .nii.gz)
#   - FreeSurfer.jl     → for MGZ
#   - Zarr.jl / HDF5.jl → for Zarr / OME-Zarr
#   - NPZ.jl            → for NPY
# ──────────────────────────────────────────────────────────────────────────────

# ── Format detection ─────────────────────────────────────────────────────────

"""
    _detect_input_format(path) → Symbol

Detect the input file format from the file path or directory structure.

# Returns
One of `:tiff`, `:tiff_dir`, `:nifti`, `:nifti_gz`, `:mgz`, `:zarr`,
`:ome_zarr`, or throws an error for unrecognised formats.
"""
function _detect_input_format(path::AbstractString)
    if isdir(path)
        # Check if it's a Zarr store
        if endswith(path, ".zarr") || isfile(joinpath(path, ".zarray"))
            if endswith(path, ".ome.zarr") || isfile(joinpath(path, ".zattrs"))
                return :ome_zarr
            end
            return :zarr
        end
        # Otherwise assume directory of TIFF slices
        return :tiff_dir
    end

    lpath = lowercase(path)
    if endswith(lpath, ".nii.gz")
        return :nifti_gz
    elseif endswith(lpath, ".nii")
        return :nifti
    elseif endswith(lpath, ".mgz") || endswith(lpath, ".mgh")
        return :mgz
    elseif endswith(lpath, ".tif") || endswith(lpath, ".tiff")
        return :tiff
    elseif endswith(lpath, ".npy")
        return :npy
    elseif endswith(lpath, ".zarr")
        return :zarr
    elseif endswith(lpath, ".ome.zarr")
        return :ome_zarr
    else
        error("Unrecognised input format for path: $path\n" *
              "Supported formats: .tif/.tiff, directory of TIFFs, .nii, .nii.gz, " *
              ".mgz/.mgh, .zarr, .ome.zarr")
    end
end

"""
    _detect_output_format(path) → Symbol

Detect the desired output format from the file extension.

# Returns
One of `:nifti`, `:nifti_gz`, `:mgz`, `:zarr`, `:ome_zarr`, `:npy`, or `:array`.
"""
function _detect_output_format(path::AbstractString)
    lpath = lowercase(path)
    if endswith(lpath, ".nii.gz")
        return :nifti_gz
    elseif endswith(lpath, ".nii")
        return :nifti
    elseif endswith(lpath, ".mgz") || endswith(lpath, ".mgh")
        return :mgz
    elseif endswith(lpath, ".ome.zarr")
        return :ome_zarr
    elseif endswith(lpath, ".zarr")
        return :zarr
    elseif endswith(lpath, ".npy")
        return :npy
    else
        error("Unrecognised output format for path: $path\n" *
              "Supported output formats: .nii, .nii.gz, .mgz, .zarr, .ome.zarr, .npy")
    end
end

# ── Loading functions ────────────────────────────────────────────────────────

"""
    load_volume(path::AbstractString) → Array{Float64, 3}

Load a 3D volume from the given file path. The format is auto-detected.

Requires the appropriate I/O package to be installed and loaded.

# Supported Formats
| Format          | Required Package  |
|:----------------|:------------------|
| TIFF file       | TiffImages.jl     |
| TIFF directory  | TiffImages.jl     |
| NIfTI (.nii)    | NIfTI.jl          |
| NIfTI (.nii.gz) | NIfTI.jl          |
| MGZ (.mgz)      | FreeSurfer.jl     |
| Zarr (.zarr)    | Zarr.jl           |
| OME-Zarr        | Zarr.jl           |
| NPY (.npy)      | NPZ.jl            |
"""
function load_volume(path::AbstractString)
    fmt = _detect_input_format(path)
    return _load_volume(Val(fmt), path)
end

# Fallback: error with installation instructions
function _load_volume(::Val{fmt}, path::AbstractString) where {fmt}
    pkg_map = Dict(
        :tiff     => "TiffImages",
        :tiff_dir => "TiffImages",
        :nifti    => "NIfTI",
        :nifti_gz => "NIfTI",
        :mgz      => "FreeSurfer",
        :zarr     => "Zarr",
        :ome_zarr => "Zarr",
        :npy      => "NPZ",
    )
    pkg = get(pkg_map, fmt, "the appropriate")
    error("Loading $fmt files requires $pkg.jl. " *
          "Install it with: using Pkg; Pkg.add(\"$pkg\")\n" *
          "Then load it with: using $pkg")
end

# Dispatch concrete loaders by format
_load_volume(::Val{:tiff},     path::AbstractString) = _load_tiff(path)
_load_volume(::Val{:tiff_dir}, path::AbstractString) = _load_tiff_dir(path)
_load_volume(::Val{:nifti},    path::AbstractString) = _load_nifti(path)
_load_volume(::Val{:nifti_gz}, path::AbstractString) = _load_nifti(path)
_load_volume(::Val{:mgz},      path::AbstractString) = _load_mgz(path)
_load_volume(::Val{:zarr},     path::AbstractString) = _load_zarr(path)
_load_volume(::Val{:ome_zarr}, path::AbstractString) = _load_zarr(path)
_load_volume(::Val{:npy},      path::AbstractString) = _load_npy(path)

# ── Concrete loaders (guarded by package availability) ───────────────────────
# These are overridden when the user loads the relevant I/O packages.
# The pattern uses @eval to define methods only when packages are available.

# TIFF (single file)
function _load_tiff(path::AbstractString)
    TiffImages = Base.require(Base.PkgId(Base.UUID("731e570b-9d59-4bfa-96dc-6df516fadf69"), "TiffImages"))
    img = TiffImages.load(path)
    return convert(Array{Float64}, img)
end

# TIFF directory (stack of 2D slices)
function _load_tiff_dir(path::AbstractString)
    TiffImages = Base.require(Base.PkgId(Base.UUID("731e570b-9d59-4bfa-96dc-6df516fadf69"), "TiffImages"))
    # Find all TIFF files, sorted naturally
    files = sort(filter(f -> occursin(r"\.(tif|tiff)$"i, f), readdir(path; join=true)))
    isempty(files) && error("No TIFF files found in directory: $path")

    # Load first slice to get dimensions
    first_slice = TiffImages.load(files[1])
    nx, ny = size(first_slice)
    nz = length(files)

    volume = Array{Float64}(undef, nx, ny, nz)
    for (i, f) in enumerate(files)
        slice = TiffImages.load(f)
        volume[:, :, i] .= convert(Array{Float64}, slice)
    end
    return volume
end

# NIfTI
function _load_nifti(path::AbstractString)
    NIfTI_mod = Base.require(Base.PkgId(Base.UUID("a3a9e032-41b5-5fc4-967a-a6b7a19844d3"), "NIfTI"))
    ni = NIfTI_mod.niread(path)
    return convert(Array{Float64}, ni.raw)
end

# MGZ (FreeSurfer)
function _load_mgz(path::AbstractString)
    FS = Base.require(Base.PkgId(Base.UUID("e6cd2a66-8224-4040-a7c3-75b97a619213"), "FreeSurfer"))
    vol = FS.load_mgh(path)
    return convert(Array{Float64}, vol.vol)
end

# Zarr
function _load_zarr(path::AbstractString)
    Zarr_mod = Base.require(Base.PkgId(Base.UUID("0a941bbe-ad1d-11e8-39d9-ab76183a1d99"), "Zarr"))
    z = Zarr_mod.zopen(path)
    return convert(Array{Float64}, z[:, :, :])
end

# NPY
function _load_npy(path::AbstractString)
    NPZ_mod = Base.require(Base.PkgId(Base.UUID("15e1cf62-19b3-5cfa-8e77-841668bca605"), "NPZ"))
    data = NPZ_mod.npzread(path)
    return convert(Array{Float64}, data)
end

# ── Saving functions ─────────────────────────────────────────────────────────

"""
    save_result(path::AbstractString, data::AbstractArray;
                header=nothing, voxel_size=nothing, affine=nothing)

Save an array to the specified output path. The format is auto-detected from
the file extension.

# Arguments
- `path`: Output file path.
- `data`: Array to save (any dimensionality).

# Keyword Arguments
- `header`: Optional NIfTI/MGZ header to preserve spatial metadata from the
  input file. If `nothing`, a default header is created.
- `voxel_size`: Optional tuple `(dx, dy, dz)` specifying voxel dimensions.
  Used when creating new NIfTI/MGZ headers.
- `affine`: Optional 3×3, 3×4, or 4×4 matrix encoding the spatial affine
  (rotation × voxel spacing, plus optional translation). Written to the NIfTI
  sform (sform_code = 1). If provided, `voxel_size` is still used for pixdim.
"""
function save_result(path::AbstractString, data::AbstractArray;
                     header = nothing,
                     voxel_size = nothing,
                     affine = nothing)
    fmt = _detect_output_format(path)
    return _save_result(Val(fmt), path, data; header = header, voxel_size = voxel_size,
                        affine = affine)
end

# Fallback
function _save_result(::Val{fmt}, path::AbstractString, data::AbstractArray;
                      header = nothing, voxel_size = nothing, affine = nothing) where {fmt}
    pkg_map = Dict(
        :nifti    => "NIfTI",
        :nifti_gz => "NIfTI",
        :mgz      => "FreeSurfer",
        :zarr     => "Zarr",
        :ome_zarr => "Zarr",
        :npy      => "NPZ",
    )
    pkg = get(pkg_map, fmt, "the appropriate")
    error("Saving $fmt files requires $pkg.jl. " *
          "Install it with: using Pkg; Pkg.add(\"$pkg\")")
end

# ── NIfTI layout helper ───────────────────────────────────────────────────────
# Neuroimaging convention: spatial dims (x,y,z) first, component dims last.
# Input arrays from this package have components leading: (c..., nx, ny, nz).
# This permutes to (nx, ny, nz, c...) and flattens >4D to 4D for compatibility
# with readers that only support up to 4 dimensions (e.g. FreeSurfer niiRead).
function _to_nifti_layout(data::AbstractArray)
    nd = ndims(data)
    nd == 3 && return data  # pure 3D — no permutation needed

    # data has shape (comp_dims..., nx, ny, nz)
    n_comp = nd - 3
    perm    = (n_comp+1:nd..., 1:n_comp...)   # spatial first, then components
    arr     = permutedims(data, perm)           # (nx, ny, nz, comp_dims...)

    # Flatten to 4D for readers that don't support >4D NIfTI
    if nd > 4
        nx, ny, nz = size(arr, 1), size(arr, 2), size(arr, 3)
        arr = reshape(arr, nx, ny, nz, :)
    end
    return arr
end

# NIfTI (.nii / .nii.gz) — niwrite auto-detects compression from extension
function _save_result(::Val{:nifti}, path::AbstractString, data::AbstractArray;
                      header = nothing, voxel_size = nothing, affine = nothing)
    NIfTI_mod = Base.require(Base.PkgId(Base.UUID("a3a9e032-41b5-5fc4-967a-a6b7a19844d3"), "NIfTI"))
    # Permute to (nx, ny, nz, components) and flatten to 4D for reader compatibility
    arr = Array{Float32}(_to_nifti_layout(data))
    ni  = voxel_size !== nothing ? NIfTI_mod.NIVolume(arr; voxel_size = voxel_size) :
                                   NIfTI_mod.NIVolume(arr)
    # Set qform_code = 1 (scanner RAS) so readers don't warn about invalid spatial form
    ni.header.qform_code = Int16(1)
    # Tag 3-frame volumes as vector fields (NIFTI_INTENT_VECTOR = 1007) so
    # FreeView / FSL expose the vector/DEC display options automatically.
    if ndims(arr) == 4 && size(arr, 4) == 3
        ni.header.intent_code = Int16(1007)
    end
    # Write spatial orientation via sform when an affine is supplied.
    # Accepts 3×3 (rotation×scale, no translation), 3×4, or 4×4.
    if affine !== nothing
        A = Float32.(affine)
        t = size(A, 2) >= 4 ? (A[1,4], A[2,4], A[3,4]) : (0f0, 0f0, 0f0)
        ni.header.sform_code = Int16(1)
        ni.header.srow_x = (A[1,1], A[1,2], A[1,3], t[1])
        ni.header.srow_y = (A[2,1], A[2,2], A[2,3], t[2])
        ni.header.srow_z = (A[3,1], A[3,2], A[3,3], t[3])
    end
    NIfTI_mod.niwrite(path, ni)
end

function _save_result(::Val{:nifti_gz}, path::AbstractString, data::AbstractArray;
                      header = nothing, voxel_size = nothing, affine = nothing)
    _save_result(Val(:nifti), path, data; header = header, voxel_size = voxel_size,
                 affine = affine)
end

# NPY — uses NPZ.jl
function _save_result(::Val{:npy}, path::AbstractString, data::AbstractArray;
                      header = nothing, voxel_size = nothing, affine = nothing)  # spatial kwargs unused for NPY
    NPZ_mod = Base.require(Base.PkgId(Base.UUID("15e1cf62-19b3-5cfa-8e77-841668bca605"), "NPZ"))
    NPZ_mod.npzwrite(path, data)
end

# ── High-level processing pipeline ──────────────────────────────────────────

"""
    process_volume(input_path, σ, ρ;
        output_path=nothing, output_format=:array,
        chunk_size=nothing, truncate=4.0,
        full=false, eigenvalue_order=:asc,
        compute_S=true, compute_eigen=true,
        verbose=false, voxel_size=nothing) → NamedTuple

End-to-end pipeline: load a volume → compute structure tensor → compute
eigendecomposition → save results.

This is the wrapper script designed for processing neuroimaging data from
various file formats without writing Julia code.

# Arguments
- `input_path::AbstractString`: Path to the input volume file or directory.
- `σ::Real`: Noise scale (inner Gaussian).
- `ρ::Real`: Integration scale (outer Gaussian).

# Keyword Arguments
- `output_dir::Union{AbstractString, Nothing} = nothing`: Directory to save outputs.
  If `nothing`, results are only returned in memory.
- `output_format::Symbol = :nifti_gz`: Output file format. One of `:nifti`,
  `:nifti_gz`, `:mgz`, `:zarr`, `:ome_zarr`, `:npy`, `:array`.
- `chunk_size::Union{Int, Nothing} = nothing`: If set, use chunked processing
  with the given block size. If `nothing`, process the whole volume at once.
- `use_gpu::Bool = false`: If `true`, use GPU out-of-core chunked processing
  (requires CUDA.jl). Forces `chunk_size` to 128 if not specified.
- `truncate::Real = 4.0`: Gaussian truncation parameter.
- `full::Bool = false`: Compute all eigenvectors (`true`) or just primary (`false`).
- `eigenvalue_order::Symbol = :asc`: Eigenvalue ordering.
- `compute_S::Bool = true`: Compute and return the structure tensor.
- `compute_eigen::Bool = true`: Compute and return eigenvalues/vectors.
- `verbose::Bool = false`: Print progress information.
- `voxel_size::Union{NTuple{3,<:Real}, Nothing} = nothing`: Voxel dimensions
  for output headers (e.g., `(0.1, 0.1, 0.1)` for 100μm isotropic).
- `affine`: Optional 3×3, 3×4, or 4×4 spatial affine matrix (rotation × voxel
  spacing + optional translation). Written to the NIfTI sform of every output.
  Build it as `R .* voxel_size'` where `R` is a 3×3 rotation matrix.
- `prefix::AbstractString = "st"`: Filename prefix for outputs.

# Returns
A `NamedTuple` with fields:
- `volume`: The loaded input volume.
- `S`: Structure tensor or `nothing`.
- `val`: Eigenvalues or `nothing`.
- `vec`: Eigenvectors or `nothing`.

# Output Files
When `output_dir` is specified, the following files are saved:
- `{prefix}_tensor.{ext}`: Structure tensor (if `compute_S`).
- `{prefix}_eigenvalues.{ext}`: Eigenvalues (if `compute_eigen`).
- `{prefix}_eigenvectors.{ext}`: Eigenvectors (if `compute_eigen`).

# Example
```julia
using StructureTensor
using NIfTI  # for NIfTI I/O

# Process a NIfTI volume and save results
results = process_volume(
    "brain.nii.gz", 1.5, 5.5;
    output_dir = "results/",
    output_format = :nifti_gz,
    chunk_size = 200,
    full = true,
    verbose = true
)

# Access results in memory
eigenvalues = results.val
eigenvectors = results.vec
```
"""
function process_volume(input_path::AbstractString,
                        σ::Real,
                        ρ::Real;
                        output_dir::Union{AbstractString, Nothing} = nothing,
                        output_format::Symbol = :nifti_gz,
                        chunk_size::Union{Int, Nothing} = nothing,
                        use_gpu::Bool = false,
                        truncate::Real = 4.0,
                        full::Bool = false,
                        eigenvalue_order::Symbol = :asc,
                        compute_S::Bool = true,
                        compute_eigen::Bool = true,
                        verbose::Bool = false,
                        voxel_size::Union{NTuple{3,<:Real}, Nothing} = nothing,
                        affine = nothing,
                        prefix::AbstractString = "st")
    # ── Step 1: Load volume ──────────────────────────────────────────────
    if verbose
        @info "Loading volume from: $input_path"
    end
    volume = load_volume(input_path)
    if verbose
        @info "  Volume size: $(size(volume)), dtype: $(eltype(volume))"
    end

    @assert ndims(volume) == 3 "Loaded volume must be 3D, got $(ndims(volume))D."

    # ── Step 2: Compute structure tensor ─────────────────────────────────
    S = nothing
    val = nothing
    vec = nothing

    if compute_S || compute_eigen
        if verbose
            @info "Computing structure tensor (σ=$σ, ρ=$ρ)..."
        end

        if use_gpu
            S = structure_tensor_3d_chunked_gpu(volume, σ, ρ;
                                                chunk_size = chunk_size !== nothing ? chunk_size : 128,
                                                truncate   = truncate,
                                                verbose    = verbose)
        elseif chunk_size !== nothing
            S = structure_tensor_3d_chunked(volume, σ, ρ;
                                            chunk_size = chunk_size,
                                            truncate   = truncate,
                                            verbose    = verbose)
        else
            S = structure_tensor_3d(volume, σ, ρ; truncate = truncate)
        end

        if verbose
            @info "  Structure tensor shape: $(size(S))"
        end
    end

    # ── Step 3: Eigendecomposition ───────────────────────────────────────
    if compute_eigen && S !== nothing
        if verbose
            @info "Computing eigendecomposition (full=$full, order=$eigenvalue_order)..."
        end

        val, vec = eig_special_3d(S; full = full, eigenvalue_order = eigenvalue_order)

        if verbose
            @info "  Eigenvalues shape: $(size(val))"
            @info "  Eigenvectors shape: $(size(vec))"
        end
    end

    # ── Step 4: Save outputs ─────────────────────────────────────────────
    if output_dir !== nothing
        mkpath(output_dir)

        # Save a NIfTI copy of the input when the source was not already NIfTI
        # (e.g. TIFF directory, Zarr, MGZ). Useful as a spatial reference.
        input_fmt = _detect_input_format(input_path)
        if input_fmt ∉ (:nifti, :nifti_gz)
            ipath = joinpath(output_dir, "$(prefix)_input.nii.gz")
            if verbose
                @info "Saving input volume as NIfTI: $ipath"
            end
            save_result(ipath, volume; voxel_size = voxel_size, affine = affine)
        end

        # Determine file extension
        ext_map = Dict(
            :nifti    => ".nii",
            :nifti_gz => ".nii.gz",
            :mgz      => ".mgz",
            :zarr     => ".zarr",
            :ome_zarr => ".ome.zarr",
            :npy      => ".npy",
        )
        ext = get(ext_map, output_format, ".nii.gz")

        if compute_S && S !== nothing
            spath = joinpath(output_dir, "$(prefix)_tensor$(ext)")
            if verbose
                @info "Saving structure tensor to: $spath"
            end
            save_result(spath, S; voxel_size = voxel_size, affine = affine)
        end

        if compute_eigen
            if val !== nothing
                vpath = joinpath(output_dir, "$(prefix)_eigenvalues$(ext)")
                if verbose
                    @info "Saving eigenvalues to: $vpath"
                end
                save_result(vpath, val; voxel_size = voxel_size, affine = affine)
            end

            if vec !== nothing
                vecpath = joinpath(output_dir, "$(prefix)_eigenvectors$(ext)")
                if verbose
                    @info "Saving eigenvectors to: $vecpath"
                end
                save_result(vecpath, vec; voxel_size = voxel_size, affine = affine)
            end
        end

        if verbose
            @info "All outputs saved to: $output_dir"
        end
    end

    return (volume = volume, S = S, val = val, vec = vec)
end
