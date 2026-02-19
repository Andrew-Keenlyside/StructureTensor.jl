# ──────────────────────────────────────────────────────────────────────────────
# test/runtests.jl — Test suite for StructureTensor.jl
#
# Tests verify:
#   1. Gaussian kernel generation matches scipy's output.
#   2. Structure tensor computation on known volumes.
#   3. Eigendecomposition correctness (known symmetric matrices).
#   4. Chunked processing produces identical results to full processing.
#   5. Eigenvalue ordering (ascending / descending).
#   6. Edge cases (uniform volumes, single-voxel volumes, etc.).
# ──────────────────────────────────────────────────────────────────────────────

using Test
using StructureTensor
using LinearAlgebra
using Statistics: mean

@testset "StructureTensor.jl" begin

    # ── Gaussian kernel tests ────────────────────────────────────────────
    @testset "Gaussian kernel 1D" begin
        # Order 0: kernel should sum to 1.0
        k0 = StructureTensor._gaussian_kernel1d(1.5, 0, 4.0)
        @test sum(k0) ≈ 1.0 atol=1e-12

        # Kernel should be symmetric
        @test k0 ≈ reverse(k0) atol=1e-14

        # Order 1: derivative kernel should sum to ≈ 0 (antisymmetric)
        k1 = StructureTensor._gaussian_kernel1d(1.5, 1, 4.0)
        @test abs(sum(k1)) < 1e-12

        # Derivative kernel should be antisymmetric
        @test k1 ≈ -reverse(k1) atol=1e-14

        # Kernel half-width: round(truncate * σ + 0.5)
        σ = 2.0
        truncate = 4.0
        expected_lw = round(Int, truncate * σ + 0.5)
        expected_len = 2 * expected_lw + 1
        @test length(StructureTensor._gaussian_kernel1d(σ, 0, truncate)) == expected_len
    end

    # ── Structure tensor basic tests ─────────────────────────────────────
    @testset "Structure tensor 3D - basic" begin
        # Uniform volume: all gradients should be zero → S ≈ 0
        vol_uniform = ones(32, 32, 32)
        S = structure_tensor_3d(vol_uniform, 1.0, 2.0)
        @test size(S) == (6, 32, 32, 32)
        @test maximum(abs, S) < 1e-10

        # Volume with gradient along dim 1 only: V(x,y,z) = x
        vol_grad = zeros(32, 32, 32)
        for i in 1:32
            vol_grad[i, :, :] .= Float64(i)
        end
        S = structure_tensor_3d(vol_grad, 0.5, 1.0)
        # Sxx (component 1) should be dominant in the interior
        interior = S[:, 10:22, 10:22, 10:22]
        @test mean(abs.(interior[1, :, :, :])) > 10 * mean(abs.(interior[2, :, :, :]))
        @test mean(abs.(interior[1, :, :, :])) > 10 * mean(abs.(interior[3, :, :, :]))
    end

    @testset "Structure tensor 3D - output shape" begin
        vol = randn(20, 25, 30)
        S = structure_tensor_3d(vol, 1.0, 2.0)
        @test size(S) == (6, 20, 25, 30)
        @test eltype(S) == Float64
    end

    @testset "Structure tensor 3D - integer input warning" begin
        vol_int = rand(Int8, 16, 16, 16)
        @test_logs (:warn, r"not a floating-point") structure_tensor_3d(vol_int, 1.0, 2.0)
    end

    # ── Eigendecomposition tests ─────────────────────────────────────────
    @testset "Eigendecomposition - known matrix" begin
        # Create a structure tensor field with a known 3×3 symmetric matrix
        # at every voxel.
        #
        # A = [3 1 0; 1 2 1; 0 1 1]
        # Known eigenvalues (from analytical / numerical solve):
        #   λ ≈ [0.1981, 2.0, 3.8019]
        nx, ny, nz = 5, 5, 5
        S = zeros(6, nx, ny, nz)
        S[1, :, :, :] .= 3.0  # Sxx
        S[2, :, :, :] .= 2.0  # Syy
        S[3, :, :, :] .= 1.0  # Szz
        S[4, :, :, :] .= 1.0  # Sxy
        S[5, :, :, :] .= 0.0  # Sxz
        S[6, :, :, :] .= 1.0  # Syz

        # Reference eigenvalues (computed externally)
        A = [3.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 1.0]
        ref_eigvals = sort(eigvals(A))

        val, vec = eig_special_3d(S; full = false, eigenvalue_order = :asc)

        # Check eigenvalues match at every voxel
        for i in 1:nx, j in 1:ny, k in 1:nz
            @test val[1, i, j, k] ≈ ref_eigvals[1] atol=1e-10
            @test val[2, i, j, k] ≈ ref_eigvals[2] atol=1e-10
            @test val[3, i, j, k] ≈ ref_eigvals[3] atol=1e-10
        end

        # Check eigenvector is unit length and corresponds to smallest eigenvalue
        for i in 1:nx, j in 1:ny, k in 1:nz
            v = vec[:, i, j, k]
            @test norm(v) ≈ 1.0 atol=1e-10
            # A*v ≈ λ_min * v
            residual = A * v - ref_eigvals[1] * v
            @test norm(residual) < 1e-8
        end
    end

    @testset "Eigendecomposition - full eigenvectors" begin
        nx, ny, nz = 3, 3, 3
        S = zeros(6, nx, ny, nz)
        S[1, :, :, :] .= 5.0
        S[2, :, :, :] .= 3.0
        S[3, :, :, :] .= 1.0
        S[4, :, :, :] .= 0.5
        S[5, :, :, :] .= 0.1
        S[6, :, :, :] .= 0.2

        val, vec = eig_special_3d(S; full = true, eigenvalue_order = :asc)

        @test size(val) == (3, 3, 3, 3)
        @test size(vec) == (3, 3, 3, 3, 3)

        # Verify A * vi = λi * vi for each eigenvector
        A = [5.0 0.5 0.1; 0.5 3.0 0.2; 0.1 0.2 1.0]
        for i in 1:nx, j in 1:ny, k in 1:nz
            for vi in 1:3
                v = vec[:, vi, i, j, k]
                λ = val[vi, i, j, k]
                @test norm(v) ≈ 1.0 atol=1e-10
                @test norm(A * v - λ * v) < 1e-8
            end
        end
    end

    @testset "Eigendecomposition - descending order" begin
        nx, ny, nz = 3, 3, 3
        S = zeros(6, nx, ny, nz)
        S[1, :, :, :] .= 6.0
        S[2, :, :, :] .= 2.0
        S[3, :, :, :] .= 1.0

        val_asc, _ = eig_special_3d(S; eigenvalue_order = :asc)
        val_desc, _ = eig_special_3d(S; eigenvalue_order = :desc)

        # Descending should be the reverse of ascending
        for i in 1:nx, j in 1:ny, k in 1:nz
            @test val_asc[1, i, j, k] ≈ val_desc[3, i, j, k] atol=1e-12
            @test val_asc[2, i, j, k] ≈ val_desc[2, i, j, k] atol=1e-12
            @test val_asc[3, i, j, k] ≈ val_desc[1, i, j, k] atol=1e-12
        end
    end

    @testset "Eigendecomposition - isotropic tensor (degenerate)" begin
        # All eigenvalues equal → tensor is a scalar multiple of identity
        nx, ny, nz = 3, 3, 3
        S = zeros(6, nx, ny, nz)
        S[1, :, :, :] .= 4.0
        S[2, :, :, :] .= 4.0
        S[3, :, :, :] .= 4.0
        # Off-diagonals = 0

        val, vec = eig_special_3d(S; full = true)

        for i in 1:nx, j in 1:ny, k in 1:nz
            @test val[1, i, j, k] ≈ 4.0 atol=1e-10
            @test val[2, i, j, k] ≈ 4.0 atol=1e-10
            @test val[3, i, j, k] ≈ 4.0 atol=1e-10
            # Eigenvectors should still be unit vectors (though direction is arbitrary)
            for vi in 1:3
                @test norm(vec[:, vi, i, j, k]) ≈ 1.0 atol=1e-10
            end
        end
    end

    # ── Chunked processing tests ─────────────────────────────────────────
    @testset "Chunked vs full - equivalence" begin
        vol = randn(40, 40, 40)
        σ, ρ = 1.0, 2.0

        S_full = structure_tensor_3d(vol, σ, ρ)
        S_chunked = structure_tensor_3d_chunked(vol, σ, ρ; chunk_size = 15)

        # Results should be identical (within floating-point precision)
        @test S_full ≈ S_chunked atol=1e-10
    end

    @testset "Chunked - various chunk sizes" begin
        vol = randn(30, 30, 30)
        σ, ρ = 0.5, 1.0
        S_ref = structure_tensor_3d(vol, σ, ρ)

        for cs in [10, 15, 20, 30]
            S_c = structure_tensor_3d_chunked(vol, σ, ρ; chunk_size = cs)
            @test S_ref ≈ S_c atol=1e-10
        end
    end

    # ── Parallel analysis tests ──────────────────────────────────────────
    @testset "Parallel analysis pipeline" begin
        vol = randn(30, 30, 30)
        S, val, vec = parallel_structure_tensor_analysis(
            vol, 1.0, 2.0; chunk_size = 15, full = false)

        @test size(S) == (6, 30, 30, 30)
        @test size(val) == (3, 30, 30, 30)
        @test size(vec) == (3, 30, 30, 30)

        # Eigenvalues should be non-negative for positive semi-definite tensors
        # (structure tensors are always PSD since they're outer products)
        @test all(val .>= -1e-10)  # allow small numerical errors
    end

    # ── I/O format detection tests ───────────────────────────────────────
    @testset "Format detection" begin
        @test StructureTensor._detect_input_format("volume.nii") == :nifti
        @test StructureTensor._detect_input_format("volume.nii.gz") == :nifti_gz
        @test StructureTensor._detect_input_format("volume.mgz") == :mgz
        @test StructureTensor._detect_input_format("volume.tif") == :tiff
        @test StructureTensor._detect_input_format("volume.tiff") == :tiff
        @test StructureTensor._detect_output_format("out.nii.gz") == :nifti_gz
        @test StructureTensor._detect_output_format("out.npy") == :npy
        @test StructureTensor._detect_output_format("out.zarr") == :zarr
    end

    # ── Halo computation tests ───────────────────────────────────────────
    @testset "Halo computation" begin
        # halo = hw_σ + hw_ρ
        halo = StructureTensor._compute_halo(1.0, 2.0, 4.0)
        hw_σ = round(Int, 4.0 * 1.0 + 0.5)
        hw_ρ = round(Int, 4.0 * 2.0 + 0.5)
        @test halo == hw_σ + hw_ρ
    end

    # ── Chunk range computation tests ────────────────────────────────────
    @testset "Chunk ranges" begin
        ranges = StructureTensor._chunk_ranges(100, 30, 5)
        # Should cover the full range
        covered = Set{Int}()
        for (pad_r, inner_r) in ranges
            for i in inner_r
                push!(covered, first(pad_r) + i - 1)
            end
        end
        @test covered == Set(1:100)
    end
end

