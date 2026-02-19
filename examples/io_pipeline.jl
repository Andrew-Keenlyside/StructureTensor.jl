# ──────────────────────────────────────────────────────────────────────────────
# examples/io_pipeline.jl — File I/O pipeline for neuroimaging data
#
# Demonstrates the `process_volume` wrapper for loading volumetric data from
# various neuroimaging formats, computing structure tensors, and saving results.
#
# Prerequisites (install whichever formats you need):
#   using Pkg
#   Pkg.add("NIfTI")        # for .nii / .nii.gz
#   Pkg.add("TiffImages")   # for TIFF files
#   Pkg.add("FreeSurfer")   # for .mgz
#   Pkg.add("Zarr")         # for Zarr / OME-Zarr
#   Pkg.add("NPZ")          # for .npy
# ──────────────────────────────────────────────────────────────────────────────

using StructureTensor

# Load the I/O packages you need:
# using NIfTI
# using TiffImages

# ═══════════════════════════════════════════════════════════════════════════════
# Example 1: NIfTI input → NIfTI output
# ═══════════════════════════════════════════════════════════════════════════════

# results = process_volume(
#     "brain_volume.nii.gz",  # input path
#     1.5,                    # σ (noise scale)
#     5.5;                    # ρ (integration scale)
#     output_dir = "results/",
#     output_format = :nifti_gz,
#     chunk_size = 200,       # chunked processing for large volumes
#     full = true,            # compute all three eigenvectors
#     verbose = true
# )

# ═══════════════════════════════════════════════════════════════════════════════
# Example 2: TIFF directory → Zarr output
# ═══════════════════════════════════════════════════════════════════════════════

# results = process_volume(
#     "path/to/tiff_slices/",  # directory of TIFF slices
#     0.5, 2.0;
#     output_dir = "results/",
#     output_format = :zarr,
#     chunk_size = 128,
#     verbose = true
# )

# ═══════════════════════════════════════════════════════════════════════════════
# Example 3: MGZ input → NPY output
# ═══════════════════════════════════════════════════════════════════════════════

# results = process_volume(
#     "brain.mgz", 1.0, 3.0;
#     output_dir = "results/",
#     output_format = :npy,
#     verbose = true
# )

# ═══════════════════════════════════════════════════════════════════════════════
# Example 4: In-memory only (no file output)
# ═══════════════════════════════════════════════════════════════════════════════

# results = process_volume("volume.nii.gz", 1.5, 5.5)
# S   = results.S     # structure tensor
# val = results.val   # eigenvalues
# vec = results.vec   # eigenvectors

println("I/O pipeline examples ready. Uncomment the format you need!")
println("Supported input formats:  TIFF, TIFF dir, NIfTI, MGZ, Zarr, OME-Zarr")
println("Supported output formats: NIfTI, NIfTI.gz, MGZ, Zarr, OME-Zarr, NPY")
