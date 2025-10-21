using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

include("PS7_Source_Subedi.jl")

@testset "PS7 Comprehensive Unit Tests" begin
    
    @testset "Data Loading Functions" begin
        @testset "load_data: structure and types" begin
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
            df, X, y = load_data(url)
            
            # Dimensions
            @test size(df, 1) > 0
            @test size(X, 2) == 4
            @test size(X, 1) == length(y)
            @test size(X, 1) == size(df, 1)
            
            # Column structure
            @test all(X[:, 1] .== 1)  # intercept
            @test all(x -> x in [0, 1], X[:, 3])  # race binary
            @test all(x -> x in [0, 1], X[:, 4])  # collgrad binary
            
            # Types
            @test eltype(X) <: AbstractFloat
            @test eltype(y) <: AbstractFloat
            @test all(isfinite.(y))
            @test !any(ismissing.(y))
        end
        
        @testset "prepare_occupation_data: categories and structure" begin
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
            df_orig, _, _ = load_data(url)
            df, X, y = prepare_occupation_data(df_orig)
            
            # Occupation collapse
            @test maximum(y) == 7
            @test minimum(y) == 1
            @test all(1 .<= y .<= 7)
            @test eltype(y) <: Integer
            
            # X structure
            @test size(X, 2) == 4
            @test all(X[:, 1] .== 1)
            @test all(x -> x in [0, 1], X[:, 2])  # white binary
            @test !any(isnan.(X))
        end
    end
    
    @testset "OLS via GMM" begin
        @testset "Objective function properties" begin
            Random.seed!(123)
            N, K = 100, 3
            X = [ones(N) randn(N, K-1)]
            β_true = [1.0, 2.0, -1.5]
            y = X * β_true + 0.1 * randn(N)
            
            obj = ols_gmm(β_true, X, y)
            
            # Basic properties
            @test isfinite(obj)
            @test obj >= 0  # GMM objective non-negative
            @test obj < 5.0  # Should be small at true params
            
            # Worse parameters give larger objective
            @test ols_gmm(zeros(K), X, y) > obj
            @test ols_gmm(10 .* ones(K), X, y) > ols_gmm(zeros(K), X, y)
        end
        
        @testset "Moment conditions" begin
            Random.seed!(456)
            N = 200
            X = [ones(N) randn(N, 2)]
            β = [0.5, 1.5, -0.8]
            y = X * β + 0.3 * randn(N)
            
            # At true parameters, moments close to zero
            ε = y - X * β
            moments = X' * ε / N
            @test maximum(abs.(moments)) < 0.15
            
            # GMM objective quadratic in moments
            obj = ols_gmm(β, X, y)
            @test obj ≈ dot(moments, moments) rtol=1e-6
        end
        
        @testset "Dimension handling" begin
            Random.seed!(789)
            
            # Small system
            X_small = [ones(20) randn(20, 1)]
            β_small = [1.0, 0.5]
            y_small = X_small * β_small + 0.1 * randn(20)
            @test ols_gmm(β_small, X_small, y_small) >= 0
            
            # Large system
            X_large = [ones(500) randn(500, 5)]
            β_large = randn(6)
            y_large = X_large * β_large + 0.1 * randn(500)
            @test ols_gmm(β_large, X_large, y_large) >= 0
        end
    end
    
    @testset "Multinomial Logit MLE" begin
        @testset "Log-likelihood properties" begin
            Random.seed!(123)
            N, K, J = 100, 3, 4
            X = [ones(N) randn(N, K-1)]
            α = randn(K * (J-1))
            y = rand(1:J, N)
            
            ll = mlogit_mle(α, X, y)
            
            # Basic properties
            @test isfinite(ll)
            @test ll < 0  # Log-likelihood always negative
            @test ll > -N * log(J) * 3  # Not unreasonably negative
        end
        
        @testset "Probability calculation" begin
            Random.seed!(456)
            N, K, J = 50, 2, 3
            X = [ones(N) randn(N, K-1)]
            α = randn(K * (J-1))
            
            # Extract probabilities manually
            α_mat = reshape(α, K, J-1)
            num = zeros(N, J)
            num[:, 1:end-1] = exp.(X * α_mat)
            num[:, end] .= 1.0
            den = sum(num, dims=2)
            P = num ./ den
            
            # Probabilities sum to 1
            @test all(abs.(sum(P, dims=2) .- 1) .< 1e-10)
            @test all(0 .<= P .<= 1)
            @test all(isfinite.(P))
        end
        
        @testset "Different specifications" begin
            Random.seed!(789)
            N = 80
            
            # Binary choice (J=2)
            X2 = [ones(N) randn(N, 1)]
            α2 = randn(2)
            y2 = rand(1:2, N)
            @test mlogit_mle(α2, X2, y2) < 0
            
            # Many alternatives (J=5)
            X5 = [ones(N) randn(N, 2)]
            α5 = randn(8)  # 2*(5-1)
            y5 = rand(1:5, N)
            @test mlogit_mle(α5, X5, y5) < 0
        end
        
        @testset "Parameter sensitivity" begin
            Random.seed!(101)
            N, K, J = 60, 3, 4
            X = [ones(N) randn(N, K-1)]
            y = rand(1:J, N)
            
            α_zero = zeros(K * (J-1))
            α_small = 0.1 .* randn(K * (J-1))
            α_large = 5.0 .* randn(K * (J-1))
            
            ll_zero = mlogit_mle(α_zero, X, y)
            ll_small = mlogit_mle(α_small, X, y)
            ll_large = mlogit_mle(α_large, X, y)
            
            # All should be valid
            @test all(isfinite.([ll_zero, ll_small, ll_large]))
            @test all([ll_zero, ll_small, ll_large] .< 0)
        end
    end
    
    @testset "Multinomial Logit GMM" begin
        @testset "Just-identified GMM" begin
            Random.seed!(123)
            N, K, J = 100, 3, 4
            X = [ones(N) randn(N, K-1)]
            α = randn(K * (J-1))
            y = rand(1:J, N)
            
            obj = mlogit_gmm(α, X, y)
            
            # Basic properties
            @test isfinite(obj)
            @test obj >= 0
            
            # Different parameters give different objectives
            obj2 = mlogit_gmm(2 .* α, X, y)
            @test obj2 != obj
        end
        
        @testset "Over-identified GMM" begin
            Random.seed!(456)
            N, K, J = 80, 3, 4
            X = [ones(N) randn(N, K-1)]
            α = randn(K * (J-1))
            y = rand(1:J, N)
            
            obj = mlogit_gmm_overid(α, X, y)
            
            # Basic properties
            @test isfinite(obj)
            @test obj >= 0
            
            # Worse parameters increase objective
            obj_bad = mlogit_gmm_overid(10 .* α, X, y)
            @test obj_bad > obj
        end
        
        @testset "Moment count verification" begin
            Random.seed!(789)
            N, K, J = 50, 2, 3
            X = [ones(N) randn(N, K-1)]
            α = randn(K * (J-1))
            
            # Just-ID: K*(J-1) moments for K*(J-1) parameters
            @test length(α) == K * (J-1)
            
            # Over-ID: N*J moments for K*(J-1) parameters
            # Should be over-identified when N*J > K*(J-1)
            @test N * J > K * (J-1)
        end
        
        @testset "Consistency across methods" begin
            Random.seed!(202)
            N, K, J = 70, 3, 4
            X = [ones(N) randn(N, K-1)]
            α = randn(K * (J-1))
            y = rand(1:J, N)
            
            # Both GMM methods should give non-negative objectives
            obj_just = mlogit_gmm(α, X, y)
            obj_over = mlogit_gmm_overid(α, X, y)
            
            @test obj_just >= 0
            @test obj_over >= 0
        end
    end
    
    @testset "Data Simulation" begin
        @testset "Inverse CDF: basic properties" begin
            Random.seed!(123)
            N, J = 1000, 4
            Y, X = sim_logit(N, J)
            
            # Dimensions
            @test length(Y) == N
            @test size(X) == (N, 4)
            
            # Y values
            @test all(1 .<= Y .<= J)
            @test eltype(Y) <: Integer
            @test length(unique(Y)) == J  # All alternatives chosen
            
            # X structure
            @test all(X[:, 1] .== 1)  # Intercept
            @test all(x -> x in [0, 1], X[:, 3])  # Binary
            @test all(0 .<= X[:, 4] .<= 10)  # Uniform[0,10]
        end
        
        @testset "Inverse CDF: distribution properties" begin
            Random.seed!(456)
            N = 5000
            Y, X = sim_logit(N, 4)
            
            # X distributions
            @test abs(mean(X[:, 2])) < 0.1  # N(0,1)
            @test abs(std(X[:, 2]) - 1) < 0.1
            @test abs(mean(X[:, 3]) - 0.5) < 0.05  # Bernoulli(0.5)
            @test abs(mean(X[:, 4]) - 5) < 0.3  # Uniform[0,10]
            
            # Choice frequencies (alternative 3 dominates)
            freq = [sum(Y .== j) / N for j in 1:4]
            @test freq[3] > 0.75
            @test freq[3] < 0.90
            @test all(freq .> 0)
            @test sum(freq) ≈ 1.0
        end
        
        @testset "Gumbel: basic properties" begin
            Random.seed!(789)
            N, J = 1000, 4
            Y, X = sim_logit_with_gumbel(N, J)
            
            # Same checks as inverse CDF
            @test length(Y) == N
            @test size(X) == (N, 4)
            @test all(1 .<= Y .<= J)
            @test all(X[:, 1] .== 1)
            @test length(unique(Y)) == J
        end
        
        @testset "Gumbel: distribution properties" begin
            Random.seed!(101)
            N = 5000
            Y, X = sim_logit_with_gumbel(N, 4)
            
            # X distributions same as inverse CDF
            @test abs(mean(X[:, 2])) < 0.1
            @test abs(std(X[:, 2]) - 1) < 0.1
            
            # Choice frequencies
            freq = [sum(Y .== j) / N for j in 1:4]
            @test freq[3] > 0.75
            @test sum(freq) ≈ 1.0
        end
        
        @testset "Method comparison" begin
            Random.seed!(303)
            N = 10000
            Y1, X1 = sim_logit(N, 4)
            Random.seed!(303)
            Y2, X2 = sim_logit_with_gumbel(N, 4)
            
            freq1 = [sum(Y1 .== j) / N for j in 1:4]
            freq2 = [sum(Y2 .== j) / N for j in 1:4]
            
            # Should produce similar distributions
            @test all(abs.(freq1 .- freq2) .< 0.03)
        end
        
        @testset "Large sample stability" begin
            Random.seed!(404)
            N_large = 50000
            Y, X = sim_logit(N_large, 4)
            
            # With large N, should converge to true probabilities
            freq = [sum(Y .== j) / N_large for j in 1:4]
            @test 0.80 < freq[3] < 0.85  # Alternative 3
            @test 0.10 < freq[1] < 0.15  # Alternative 1
            @test freq[2] < 0.05  # Alternative 2
            @test freq[4] < 0.05  # Alternative 4
        end
        
        @testset "Different J values" begin
            Random.seed!(505)
            
            # J=2 (binary)
            Y2, X2 = sim_logit(500, 2)
            @test all(1 .<= Y2 .<= 2)
            @test length(unique(Y2)) == 2
            
            # J=6 (many alternatives)
            Y6, X6 = sim_logit(1000, 6)
            @test all(1 .<= Y6 .<= 6)
            @test length(unique(Y6)) <= 6
        end
    end
    
    @testset "SMM Estimation" begin
        @testset "Basic functionality" begin
            Random.seed!(123)
            N, K, J, D = 100, 3, 4, 10
            X = [ones(N) randn(N, K-1)]
            α = randn(K * (J-1))
            y = rand(1:J, N)
            
            obj = mlogit_smm_overid(α, X, y, D)
            
            @test isfinite(obj)
            @test obj >= 0
        end
        
        @testset "Draw count sensitivity" begin
            Random.seed!(456)
            N, K, J = 80, 3, 4
            X = [ones(N) randn(N, K-1)]
            α = randn(K * (J-1))
            y = rand(1:J, N)
            
            obj_5 = mlogit_smm_overid(α, X, y, 5)
            obj_50 = mlogit_smm_overid(α, X, y, 50)
            
            # Both valid
            @test isfinite(obj_5) && isfinite(obj_50)
            @test obj_5 >= 0 && obj_50 >= 0
            
            # More draws generally more stable (smaller variance)
            # Run multiple times
            objs_5 = [mlogit_smm_overid(α, X, y, 5) for _ in 1:5]
            objs_50 = [mlogit_smm_overid(α, X, y, 50) for _ in 1:5]
            
            @test std(objs_50) < std(objs_5) * 2  # More draws = less variable
        end
        
        @testset "Parameter sensitivity" begin
            Random.seed!(789)
            N, K, J, D = 60, 2, 3, 20
            X = [ones(N) randn(N, K-1)]
            y = rand(1:J, N)
            
            α_zero = zeros(K * (J-1))
            α_small = 0.1 .* randn(K * (J-1))
            
            obj_zero = mlogit_smm_overid(α_zero, X, y, D)
            obj_small = mlogit_smm_overid(α_small, X, y, D)
            
            @test obj_zero != obj_small
            @test all(isfinite.([obj_zero, obj_small]))
        end
        
        @testset "Consistency with GMM" begin
            Random.seed!(101)
            N, K, J = 100, 3, 4
            X = [ones(N) randn(N, K-1)]
            α = randn(K * (J-1))
            y = rand(1:J, N)
            
            # SMM and GMM should both be non-negative
            obj_gmm = mlogit_gmm_overid(α, X, y)
            obj_smm = mlogit_smm_overid(α, X, y, 100)
            
            @test obj_gmm >= 0
            @test obj_smm >= 0
        end
    end
    
    @testset "Edge Cases and Robustness" begin
        @testset "Small sample behavior" begin
            Random.seed!(606)
            N_small = 20
            
            X = [ones(N_small) randn(N_small, 2)]
            β = [1.0, 0.5, -0.5]
            y = X * β + 0.1 * randn(N_small)
            
            @test ols_gmm(β, X, y) >= 0
            @test isfinite(ols_gmm(β, X, y))
        end
        
        @testset "Perfect multicollinearity handling" begin
            Random.seed!(707)
            N = 50
            
            # Create collinear X
            X_base = randn(N, 2)
            X = [ones(N) X_base X_base[:, 1]]  # Last column = first predictor
            
            # Functions should still compute (though results may be unstable)
            β = randn(4)
            y = randn(N)
            @test isfinite(ols_gmm(β, X, y))
        end
        
        @testset "Extreme parameter values" begin
            Random.seed!(808)
            N, K, J = 50, 3, 4
            X = [ones(N) randn(N, K-1)]
            y = rand(1:J, N)
            
            # Very large parameters
            α_large = 100.0 .* randn(K * (J-1))
            @test isfinite(mlogit_mle(α_large, X, y))
            
            # Very small parameters
            α_tiny = 1e-6 .* randn(K * (J-1))
            @test isfinite(mlogit_mle(α_tiny, X, y))
        end
        
        @testset "All same choice" begin
            Random.seed!(909)
            N, K, J = 50, 3, 4
            X = [ones(N) randn(N, K-1)]
            y = ones(Int, N)  # Everyone chooses alternative 1
            α = randn(K * (J-1))
            
            # Should still compute (log-likelihood will be poor)
            @test isfinite(mlogit_mle(α, X, y))
            @test mlogit_mle(α, X, y) < 0
        end
    end
end

println("\n✓ All comprehensive tests passed!")
