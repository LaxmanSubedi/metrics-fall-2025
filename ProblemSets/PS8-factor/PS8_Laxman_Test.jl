using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, LineSearches, MultivariateStats, FreqTables, ForwardDiff

# Set working directory
cd(@__DIR__)

include("PS8_Laxman_Source.jl")

@testset "PS8 Factor Model Tests" begin

    #==========================================================================
    # Question 1: Data Loading Tests
    ==========================================================================#
    @testset "Question 1: Data Loading and Base Regression" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        
        @testset "load_data() function" begin
            df = load_data(url)
            
            # Test return type
            @test df isa DataFrame
            
            # Test dimensions
            @test size(df, 1) > 0  # Has rows
            @test size(df, 2) == 13  # Expected 13 columns
            
            # Test required columns exist
            required_cols = ["logwage", "black", "hispanic", "female", "schoolt", 
                           "gradHS", "grad4yr", "asvabAR", "asvabCS", "asvabMK", 
                           "asvabNO", "asvabPC", "asvabWK"]
            for col in required_cols
                @test col in names(df)
            end
            
            # Test no missing values in key columns
            @test !any(ismissing.(df.logwage))
            @test !any(ismissing.(df.black))
            
            # Test data types
            @test eltype(df.logwage) <: Real
            @test eltype(df.black) <: Real
        end
    end

    #==========================================================================
    # Question 2: ASVAB Correlations Tests
    ==========================================================================#
    @testset "Question 2: ASVAB Correlations" begin
        # Create test data
        Random.seed!(123)
        n = 100
        test_df = DataFrame(
            logwage = randn(n),
            black = rand(0:1, n),
            hispanic = rand(0:1, n),
            female = rand(0:1, n),
            schoolt = rand(8:16, n),
            gradHS = rand(0:1, n),
            grad4yr = rand(0:1, n),
            asvabAR = randn(n),
            asvabCS = randn(n),
            asvabMK = randn(n),
            asvabNO = randn(n),
            asvabPC = randn(n),
            asvabWK = randn(n)
        )
        
        @testset "compute_asvab_correlations() function" begin
            cordf = compute_asvab_correlations(test_df)
            
            # Test return type
            @test cordf isa DataFrame
            
            # Test dimensions (6x6 correlation matrix)
            @test size(cordf) == (6, 6)
            
            # Test column names
            @test names(cordf) == ["cor1", "cor2", "cor3", "cor4", "cor5", "cor6"]
            
            # Convert to matrix for further tests
            cor_mat = Matrix(cordf)
            
            # Test diagonal elements are 1
            @test all(isapprox.(diag(cor_mat), 1.0, atol=1e-10))
            
            # Test symmetry
            @test isapprox(cor_mat, cor_mat', atol=1e-10)
            
            # Test all correlations are between -1 and 1
            @test all(-1 .<= cor_mat .<= 1)
            
            # Test no NaN values
            @test !any(isnan.(cor_mat))
        end
    end

    #==========================================================================
    # Question 4: PCA Tests
    ==========================================================================#
    @testset "Question 4: PCA Regression" begin
        Random.seed!(123)
        n = 100
        test_df = DataFrame(
            logwage = randn(n),
            black = rand(0:1, n),
            hispanic = rand(0:1, n),
            female = rand(0:1, n),
            schoolt = rand(8:16, n),
            gradHS = rand(0:1, n),
            grad4yr = rand(0:1, n),
            asvabAR = randn(n),
            asvabCS = randn(n),
            asvabMK = randn(n),
            asvabNO = randn(n),
            asvabPC = randn(n),
            asvabWK = randn(n)
        )
        
        @testset "generate_pca!() function" begin
            df_result = generate_pca!(test_df)
            
            # Test return type
            @test df_result isa DataFrame
            
            # Test new column was added
            @test "asvabPCA" in names(df_result)
            
            # Test correct length
            @test length(df_result.asvabPCA) == n
            
            # Test no NaN values
            @test !any(isnan.(df_result.asvabPCA))
            
            # Test PCA scores have mean approximately 0
            @test isapprox(mean(df_result.asvabPCA), 0.0, atol=0.1)
            
            # Test variance is positive
            @test var(df_result.asvabPCA) > 0
            
            # Test original columns are preserved
            @test size(df_result, 2) == size(test_df, 2) + 1
        end
    end

    #==========================================================================
    # Question 5: Factor Analysis Tests
    ==========================================================================#
    @testset "Question 5: Factor Analysis Regression" begin
        Random.seed!(123)
        n = 100
        test_df = DataFrame(
            logwage = randn(n),
            black = rand(0:1, n),
            hispanic = rand(0:1, n),
            female = rand(0:1, n),
            schoolt = rand(8:16, n),
            gradHS = rand(0:1, n),
            grad4yr = rand(0:1, n),
            asvabAR = randn(n),
            asvabCS = randn(n),
            asvabMK = randn(n),
            asvabNO = randn(n),
            asvabPC = randn(n),
            asvabWK = randn(n)
        )
        
        @testset "generate_factor!() function" begin
            df_result = generate_factor!(test_df)
            
            # Test return type
            @test df_result isa DataFrame
            
            # Test new column was added
            @test "asvabFactor" in names(df_result)
            
            # Test correct length
            @test length(df_result.asvabFactor) == n
            
            # Test no NaN values
            @test !any(isnan.(df_result.asvabFactor))
            
            # Test variance is positive
            @test var(df_result.asvabFactor) > 0
            
            # Test original columns are preserved
            @test size(df_result, 2) == size(test_df, 2) + 1
        end
    end

    #==========================================================================
    # Question 6: Factor Model MLE Tests
    ==========================================================================#
    @testset "Question 6: Factor Model MLE" begin
        Random.seed!(123)
        n = 50  # Smaller sample for faster tests
        test_df = DataFrame(
            logwage = randn(n),
            black = rand(0:1, n),
            hispanic = rand(0:1, n),
            female = rand(0:1, n),
            schoolt = rand(8:16, n),
            gradHS = rand(0:1, n),
            grad4yr = rand(0:1, n),
            asvabAR = randn(n),
            asvabCS = randn(n),
            asvabMK = randn(n),
            asvabNO = randn(n),
            asvabPC = randn(n),
            asvabWK = randn(n)
        )
        
        @testset "prepare_factor_matrices() function" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
            
            # Test dimensions
            @test size(X) == (n, 7)  # N x K (7 covariates including constant)
            @test length(y) == n
            @test size(Xfac) == (n, 4)  # N x L (4 covariates including constant)
            @test size(asvabs) == (n, 6)  # N x J (6 ASVAB tests)
            
            # Test no NaN values
            @test !any(isnan.(X))
            @test !any(isnan.(y))
            @test !any(isnan.(Xfac))
            @test !any(isnan.(asvabs))
            
            # Test constant columns (last column should be all 1s)
            @test all(X[:, end] .== 1.0)
            @test all(Xfac[:, end] .== 1.0)
            
            # Test data types
            @test X isa Matrix{<:Real}
            @test y isa Vector{<:Real}
            @test Xfac isa Matrix{<:Real}
            @test asvabs isa Matrix{<:Real}
        end
        
        @testset "factor_model() likelihood computation" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
            
            # Create parameter vector
            L = size(Xfac, 2)  # 4
            J = size(asvabs, 2)  # 6
            K = size(X, 2)  # 7
            
            # Create starting values
            θ = vcat(
                0.1 * randn(L * J),  # γ parameters (L×J = 24)
                0.1 * randn(K),       # β parameters (K = 7)
                0.1 * randn(J + 1),   # α parameters (J+1 = 7)
                abs.(0.5 .+ 0.1*randn(J + 1))  # σ parameters (J+1 = 7), ensure positive
            )
            
            # Test likelihood computation
            negloglike = factor_model(θ, X, Xfac, asvabs, y, 5)
            
            # Test return type and properties
            @test negloglike isa Real
            @test isfinite(negloglike)
            @test negloglike > 0  # Negative log-likelihood should be positive
            
            # Test with different quadrature points
            negloglike_r3 = factor_model(θ, X, Xfac, asvabs, y, 3)
            negloglike_r7 = factor_model(θ, X, Xfac, asvabs, y, 7)
            
            @test isfinite(negloglike_r3)
            @test isfinite(negloglike_r7)
        end
        
        @testset "run_estimation() function structure" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
            
            L = size(Xfac, 2)
            J = size(asvabs, 2)
            K = size(X, 2)
            
            # Create reasonable starting values
            start_vals = vcat(
                vec(Xfac \ asvabs),   # γ from OLS
                X \ y,                 # β from OLS
                0.1 * ones(J + 1),    # α small positive
                0.5 * ones(J + 1)     # σ starting at 0.5
            )
            
            # Test that starting values have correct length
            expected_length = L*J + K + (J+1) + (J+1)
            @test length(start_vals) == expected_length
            
            # Test all starting values are finite
            @test all(isfinite.(start_vals))
            
            # Note: Full optimization test is commented out as it may take time
            # Uncomment to test full estimation:
            # θ̂, se, loglike = run_estimation(test_df, start_vals)
            # @test length(θ̂) == length(start_vals)
            # @test length(se) == length(θ̂)
            # @test all(isfinite.(θ̂))
            # @test all(isfinite.(se))
            # @test all(se .> 0)
        end
    end

    #==========================================================================
    # Edge Cases and Robustness Tests
    ==========================================================================#
    @testset "Edge Cases and Robustness" begin
        
        @testset "PCA with highly correlated data" begin
            Random.seed!(456)
            n = 100
            # Create highly correlated ASVAB scores
            base = randn(n)
            test_df = DataFrame(
                logwage = randn(n),
                black = rand(0:1, n),
                hispanic = rand(0:1, n),
                female = rand(0:1, n),
                schoolt = rand(8:16, n),
                gradHS = rand(0:1, n),
                grad4yr = rand(0:1, n),
                asvabAR = base + 0.1*randn(n),
                asvabCS = base + 0.1*randn(n),
                asvabMK = base + 0.1*randn(n),
                asvabNO = base + 0.1*randn(n),
                asvabPC = base + 0.1*randn(n),
                asvabWK = base + 0.1*randn(n)
            )
            
            # Should handle highly correlated data
            @test_nowarn begin
                df_pca = generate_pca!(test_df)
                @test "asvabPCA" in names(df_pca)
            end
        end
        
        @testset "Factor Analysis with highly correlated data" begin
            Random.seed!(456)
            n = 100
            base = randn(n)
            test_df = DataFrame(
                logwage = randn(n),
                black = rand(0:1, n),
                hispanic = rand(0:1, n),
                female = rand(0:1, n),
                schoolt = rand(8:16, n),
                gradHS = rand(0:1, n),
                grad4yr = rand(0:1, n),
                asvabAR = base + 0.1*randn(n),
                asvabCS = base + 0.1*randn(n),
                asvabMK = base + 0.1*randn(n),
                asvabNO = base + 0.1*randn(n),
                asvabPC = base + 0.1*randn(n),
                asvabWK = base + 0.1*randn(n)
            )
            
            # Should handle highly correlated data
            @test_nowarn begin
                df_fa = generate_factor!(test_df)
                @test "asvabFactor" in names(df_fa)
            end
        end
        
        @testset "Correlation matrix properties" begin
            Random.seed!(789)
            n = 100
            test_df = DataFrame(
                logwage = randn(n),
                black = rand(0:1, n),
                hispanic = rand(0:1, n),
                female = rand(0:1, n),
                schoolt = rand(8:16, n),
                gradHS = rand(0:1, n),
                grad4yr = rand(0:1, n),
                asvabAR = randn(n),
                asvabCS = randn(n),
                asvabMK = randn(n),
                asvabNO = randn(n),
                asvabPC = randn(n),
                asvabWK = randn(n)
            )
            
            cordf = compute_asvab_correlations(test_df)
            cor_mat = Matrix(cordf)
            
            # Test positive semi-definiteness
            eigenvalues = eigvals(cor_mat)
            @test all(eigenvalues .> -1e-10)  # Allow small numerical errors
            
            # Test that it's a valid correlation matrix
            @test isapprox(maximum(abs.(cor_mat)), 1.0, atol=1e-10)
        end
    end

    #==========================================================================
    # Parameter Unpacking Tests for factor_model
    ==========================================================================#
    @testset "Parameter Structure Tests" begin
        Random.seed!(999)
        n = 30
        test_df = DataFrame(
            logwage = randn(n),
            black = rand(0:1, n),
            hispanic = rand(0:1, n),
            female = rand(0:1, n),
            schoolt = rand(8:16, n),
            gradHS = rand(0:1, n),
            grad4yr = rand(0:1, n),
            asvabAR = randn(n),
            asvabCS = randn(n),
            asvabMK = randn(n),
            asvabNO = randn(n),
            asvabPC = randn(n),
            asvabWK = randn(n)
        )
        
        X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
        L, J, K = size(Xfac, 2), size(asvabs, 2), size(X, 2)
        
        @testset "Parameter vector dimensions" begin
            # Total parameters should be: L*J + K + (J+1) + (J+1)
            total_params = L*J + K + (J+1) + (J+1)
            
            θ = randn(total_params)
            
            # Test that factor_model accepts this parameter vector
            @test_nowarn factor_model(θ, X, Xfac, asvabs, y, 5)
            
            # Test incorrect parameter length throws error or gives inf
            θ_wrong = randn(total_params - 1)
            @test_throws Exception factor_model(θ_wrong, X, Xfac, asvabs, y, 5)
        end
    end

end

println("\n" * "="^80)
println("All tests completed!")
println("="^80)