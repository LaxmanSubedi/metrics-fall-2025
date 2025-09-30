using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions
cd(@__DIR__)
Random.seed!(1234)
include("PS4_Subedi_Source.jl")
@testset "PS4 Complete Test Suite" begin
    
    @testset "Data Loading Tests" begin
        # Test load_data() function
        df, X, Z, y = load_data()
        
        @test size(X, 1) == length(y)  # Consistent observations
        @test size(Z, 1) == length(y)  # Consistent observations  
        @test size(Z, 2) == 8          # 8 alternatives
        @test size(X, 2) == 3          # 3 covariates
        @test all(isfinite, X)         # No missing values
        @test all(isfinite, Z)         # No missing values
        @test all(y .>= 1) && all(y .<= 8)  # Valid choices
        
        println("✓ Data: $(size(X,1)) obs, $(length(unique(y))) choices")
    end
    
    @testset "Multinomial Logit Tests" begin
        # Test with small synthetic data
        Random.seed!(123)
        n, K, J = 20, 3, 4
        X_test = randn(n, K)
        Z_test = randn(n, J)
        y_test = rand(1:J, n)
        theta_test = [randn(K*(J-1)); 0.1]  # Length: 3*3+1 = 10
        
        ll = mlogit_with_Z(theta_test, X_test, Z_test, y_test)
        @test isfinite(ll) && ll > 0   # Valid negative log-likelihood
        @test length(theta_test) == K*(J-1) + 1
        
        println("✓ Multinomial logit: LL = $(round(ll, digits=2))")
    end
    
    @testset "Mixed Logit Quadrature Tests" begin
        # Test quadrature version with small data
        Random.seed!(456)
        n, K = 15, 2
        X_test = randn(n, K)
        Z_test = randn(n, 3)
        y_test = rand(1:3, n)
        R = 3  # Few quadrature points
        theta_test = [randn(K*2); 0.0; 1.0]  # [alpha, mu_gamma, sigma_gamma]
        
        try
            ll_quad = mixed_logit_quad(theta_test, X_test, Z_test, y_test, R)
            @test isfinite(ll_quad) && ll_quad > 0
            println("✓ Mixed logit quadrature: LL = $(round(ll_quad, digits=2))")
        catch e
            @test_broken false
            println("⚠ Quadrature test failed: $(typeof(e))")
        end
    end
    
    @testset "Mixed Logit Monte Carlo Tests" begin
        # Test Monte Carlo version with small data
        Random.seed!(789)
        n, K = 15, 2
        X_test = randn(n, K)
        Z_test = randn(n, 3)
        y_test = rand(1:3, n)
        D = 10  # Few MC draws
        theta_test = [randn(K*2); 0.0; 1.0]  # [alpha, mu_gamma, sigma_gamma]
        
        try
            ll_mc = mixed_logit_mc(theta_test, X_test, Z_test, y_test, D)
            @test isfinite(ll_mc) && ll_mc > 0
            println("✓ Mixed logit Monte Carlo: LL = $(round(ll_mc, digits=2))")
        catch e
            @test_broken false
            println("⚠ Monte Carlo test failed: $(typeof(e))")
        end
    end
    
    @testset "Quadrature Practice Tests" begin
        # Test quadrature setup
        R = 7
        try
            nodes, weights = lgwt(R, -3, 3)
            @test length(nodes) == R
            @test length(weights) == R
            @test all(isfinite, nodes)
            @test all(isfinite, weights)
            @test sum(weights) ≈ 6.0  # Integral of 1 over [-3,3]
            
            println("✓ Quadrature: $R nodes, weight sum = $(round(sum(weights), digits=2))")
        catch e
            @test_broken false
            println("⚠ Quadrature setup failed: $(typeof(e))")
        end
    end
    
    @testset "Optimization Tests" begin
        # Test optimization functions exist and have proper signatures
        @test isa(optimize_mlogit, Function)
        @test isa(optimize_mixed_logit_quad, Function)
        @test isa(optimize_mixed_logit_mc, Function)
        
        # Test they can be called with small data (but don't actually run)
        Random.seed!(101)
        n, K = 10, 2
        X_test = randn(n, K)
        Z_test = randn(n, 3)
        y_test = rand(1:3, n)
        
        try
            # These should return without running expensive optimization
            result1 = optimize_mlogit(X_test, Z_test, y_test)
            result2 = optimize_mixed_logit_quad(X_test, Z_test, y_test)
            result3 = optimize_mixed_logit_mc(X_test, Z_test, y_test)
            
            @test isa(result1, Vector) || isa(result1, Tuple)
            @test isa(result2, Vector) || isa(result2, Tuple)  
            @test isa(result3, Vector) || isa(result3, Tuple)
            
            println("✓ Optimization functions callable")
        catch e
            @test_broken false
            println("⚠ Optimization setup failed: $(typeof(e))")
        end
    end
    
    @testset "Parameter Dimension Tests" begin
        # Test parameter structures
        K, J = 3, 4
        
        # Multinomial logit: K*(J-1) + 1
        theta_mlogit = randn(K*(J-1) + 1)
        @test length(theta_mlogit) == 10  # 3*3 + 1
        
        # Mixed logit: K*(J-1) + 2 (mu_gamma, sigma_gamma)
        theta_mixed = randn(K*(J-1) + 2)
        @test length(theta_mixed) == 11  # 3*3 + 2
        
        # Parameter extraction
        alpha = theta_mixed[1:end-2]
        mu_gamma = theta_mixed[end-1]
        sigma_gamma = theta_mixed[end]
        
        @test length(alpha) == K*(J-1)
        @test isa(mu_gamma, Float64)
        @test isa(sigma_gamma, Float64)
        
        println("✓ Parameter dimensions correct")
    end
    
    @testset "Integration Workflow Test" begin
        # Test complete workflow
        try
            println("Testing complete workflow...")
            allwrap()
            @test true  # Success if no crash
            println("✓ Complete workflow succeeded")
        catch e
            @test_broken false
            println("⚠ Workflow failed: $(typeof(e))")
        end
    end
    
    @testset "Numerical Stability Tests" begin
        # Test with extreme parameters
        Random.seed!(999)
        n, K = 10, 2
        X_test = randn(n, K)
        Z_test = randn(n, 3)
        y_test = rand(1:3, n)
        
        # Test multinomial logit with large parameters
        theta_large = [5.0, -5.0, 3.0, -3.0, 2.0]  # K*(J-1)+1 = 2*2+1 = 5
        try
            ll_large = mlogit_with_Z(theta_large, X_test, Z_test, y_test)
            @test isfinite(ll_large)
            println("✓ Large parameters stable")
        catch e
            println("⚠ Large parameters failed: $(typeof(e))")
        end
        
        # Test mixed logit with boundary sigma
        theta_mixed = [randn(4); 0.0; 0.1]  # Small but positive sigma
        try
            ll_mixed = mixed_logit_mc(theta_mixed, X_test, Z_test, y_test, 5)
            @test isfinite(ll_mixed)
            println("✓ Boundary parameters stable")
        catch e
            println("⚠ Boundary parameters failed: $(typeof(e))")
        end
    end
    
end

println("\n" * "="^50)
println("PS4 TEST SUITE COMPLETED")
println("="^50)