using StructureTensor
using CUDA
using TiffImages
using NIfTI

# ── Parameters ────────────────────────────────────────────────────────────────

const INPUT_DIR  = "/hdd/data/sensitivity_chunk/sensitivity_analysis_chunk_512_8-4-1"
const OUTPUT_DIR = "/hdd/data/results"

const σ          = 6     # noise scale (inner Gaussian)
const ρ          = 12    # integration scale (outer Gaussian)
const CHUNK_SIZE = 128   # voxels per chunk edge (tune to GPU VRAM)

# ── Spatial metadata ──────────────────────────────────────────────────────────
# Voxel size in mm (or whatever physical unit your data uses).
const VOXEL_SIZE = (0.2f0, 0.2f0, 0.2f0)   # isotropic 200 µm

# 3×3 rotation matrix describing the data's orientation in scanner/world space.
# Identity = standard RAS orientation; replace with your actual rotation if the
# data was acquired at an angle.
const ROTATION = Float32[1  0  0;
                          0  1  0;
                          0  0  1]

# NIfTI sform affine: M[i,j] = R[i,j] * voxel_size[j], no translation.
# Adjust the last column (zeros) to set an origin offset if needed.
const AFFINE = hcat(ROTATION .* collect(VOXEL_SIZE)', zeros(Float32, 3, 1))

# ── FA options ────────────────────────────────────────────────────────────────
const COMPUTE_FA      = true    # compute and save FA map from eigenvalues
const COMPUTE_FA_MASK = true    # also save a binary FA mask
const FA_THRESHOLD    = 0.2f0   # voxels with FA > threshold are foreground

# ── Run ───────────────────────────────────────────────────────────────────────

results = process_volume(
    INPUT_DIR,
    σ, ρ;
    output_dir    = OUTPUT_DIR,
    output_format = :nifti_gz,
    use_gpu       = true,
    chunk_size    = CHUNK_SIZE,
    full          = false,      # primary eigenvector only → (3, nx, ny, nz)
    compute_S     = false,      # skip saving the 6-component tensor
    verbose       = true,
    voxel_size    = VOXEL_SIZE,
    affine        = AFFINE,
)

# results.val  — eigenvalues  (3, nx, ny, nz)  λ1 ≤ λ2 ≤ λ3
# results.vec  — eigenvectors (3, nx, ny, nz)  primary eigenvector (min λ)

# ── FA map ────────────────────────────────────────────────────────────────────
# FA = sqrt(3/2) * std(λ) / rms(λ), ranging 0 (isotropic) → 1 (anisotropic)
if COMPUTE_FA && results.val !== nothing
    λ1 = results.val[1,:,:,:]
    λ2 = results.val[2,:,:,:]
    λ3 = results.val[3,:,:,:]
    λ_mean = (λ1 .+ λ2 .+ λ3) ./ 3
    num = sqrt(3/2) .* sqrt.((λ1 .- λ_mean).^2 .+
                              (λ2 .- λ_mean).^2 .+
                              (λ3 .- λ_mean).^2)
    den = sqrt.(λ1.^2 .+ λ2.^2 .+ λ3.^2)
    fa  = clamp.(num ./ max.(den, eps(Float64)), 0.0, 1.0)

    save_result(joinpath(OUTPUT_DIR, "st_fa.nii.gz"), fa;
                voxel_size = VOXEL_SIZE, affine = AFFINE)

    if COMPUTE_FA_MASK
        fa_mask = Float32.(fa .> FA_THRESHOLD)
        save_result(joinpath(OUTPUT_DIR, "st_fa_mask.nii.gz"), fa_mask;
                    voxel_size = VOXEL_SIZE, affine = AFFINE)
    end
end

# julia --project=/home/andrew/scripts/StructureTensor.jl  /home/andrew/scripts/StructureTensor.jl/scripts/run_structure_tensor.jl
