using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
cd(@__DIR__)
include("PS5_Source_subedi.jl")
@testset "Bus Engine Replacement Model Tests" begin
    
    # Test 1: Data Loading and Structure
    @testset "Data Loading Tests" begin
        df_long = load_static_data()
        d = load_dynamic_data()
        
        # Static data tests
        @test isa(df_long, DataFrame)
        @test all(df_long.Y .∈ Ref([0, 1]))
        @test all(df_long.Odometer .>= 0)
        @test :bus_id in names(df_long) && :Y in names(df_long)
        
        # Dynamic data tests
        @test haskey(d, :Y) && haskey(d, :X) && haskey(d, :xtran)
        @test size(d.Y) == (d.N, d.T)
        @test all(d.Y .∈ Ref([0, 1])) && all(d.X .>= 0)
        @test 0 < d.β < 1
        @test all(sum(d.xtran, dims=2) .≈ 1.0) # Transition probabilities
        
        println("✓ Data loading tests passed")
    end
    
    # Test 2: Static Model
    @testset "Static Model Tests" begin
        df_long = load_static_data()
        result = estimate_static_model(df_long)
        
        @test isa(result, StatsModels.TableRegressionModel)
        @test length(coef(result)) == 3
        @test all(isfinite.(coef(result)))
        @test coef(result)[2] < 0 # Mileage coefficient should be negative
        
        println("✓ Static model tests passed - Coefficients: ", round.(coef(result), digits=4))
    end
    
    # Test 3: Future Value Function
    @testset "Future Value Tests" begin
        d = load_dynamic_data()
        θ_test = [2.0, -0.1, 1.0]
        FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
        
        compute_future_value!(FV, θ_test, d)
        
        # Structure tests
        @test size(FV) == (d.zbin * d.xbin, 2, d.T + 1)
        @test all(FV[:, :, d.T + 1] .== 0) # Terminal condition
        @test all(isfinite.(FV))
        
        # Economic intuition: values should generally decrease over time
        @test mean(FV[:, 1, 1]) >= mean(FV[:, 1, d.T])
        
        println("✓ Future value tests passed")
    end
    
    # Test 4: Log Likelihood Function
    @testset "Likelihood Tests" begin
        d = load_dynamic_data()
        θ_test = [2.0, -0.1, 1.0]
        
        ll = log_likelihood_dynamic(θ_test, d)
        @test isfinite(ll) && ll > 0 # Should be positive (negative log-likelihood)
        
        # Different parameters should give different likelihoods
        ll2 = log_likelihood_dynamic([1.5, -0.2, 0.5], d)
        @test ll != ll2
        
        # Numerical gradient check
        ε = 1e-6
        for i in 1:3
            θ_plus = copy(θ_test); θ_plus[i] += ε
            θ_minus = copy(θ_test); θ_minus[i] -= ε
            grad = (log_likelihood_dynamic(θ_plus, d) - log_likelihood_dynamic(θ_minus, d)) / (2ε)
            @test isfinite(grad)
        end
        
        println("✓ Likelihood tests passed - LL value: ", round(ll, digits=2))
    end
    
    # Test 5: Optimization Setup
    @testset "Optimization Tests" begin
        d = load_dynamic_data()
        
        # Create mini dataset for testing
        d_mini = (
            Y = d.Y[1:5, 1:3], X = d.X[1:5, 1:3], B = d.B[1:5],
            Xstate = d.Xstate[1:5, 1:3], Zstate = d.Zstate[1:5],
            N = 5, T = 3, xval = d.xval, xbin = d.xbin, zbin = d.zbin,
            xtran = d.xtran, β = d.β
        )
        
        θ_start = [1.0, -0.1, 0.5]
        ll = log_likelihood_dynamic(θ_start, d_mini)
        @test isfinite(ll)
        
        println("✓ Optimization setup tests passed")
    end
    
    # Test 6: Edge Cases and Robustness
    @testset "Robustness Tests" begin
        d = load_dynamic_data()
        
        # Extreme parameter values
        @test isfinite(log_likelihood_dynamic([5.0, 0.0, 5.0], d))
        @test isfinite(log_likelihood_dynamic([-5.0, -0.5, -5.0], d))
        
        # Data consistency
        @test all(1 .<= d.Xstate .<= d.xbin)
        @test all(1 .<= d.Zstate .<= d.zbin)
        
        println("✓ Robustness tests passed")
    end
    
    # Test 7: Performance Check
    @testset "Performance Tests" begin
        d = load_dynamic_data()
        θ_test = [2.0, -0.1, 1.0]
        
        elapsed_time = @elapsed log_likelihood_dynamic(θ_test, d)
        @test elapsed_time < 60.0 # Should complete within 60 seconds
        
        println("✓ Performance tests passed - Time: $(round(elapsed_time, digits=3))s")
    end
    
    # Test 8: Integration Test
    @testset "Integration Tests" begin
        # End-to-end pipeline
        df_long = load_static_data()
        static_result = estimate_static_model(df_long)
        
        d = load_dynamic_data()
        ll = log_likelihood_dynamic([1.0, -0.1, 0.5], d)
        
        @test isa(static_result, StatsModels.TableRegressionModel)
        @test isfinite(ll)
        
        println("✓ Integration tests passed")
    end
end

# Utility functions
function run_all_tests()
    println("\n" * "="^50)
    println("RUNNING COMPREHENSIVE TEST SUITE")
    println("="^50)
    @time include("PS5_Test_subedi.jl")
    println("\n" * "="^50)
    println("ALL TESTS COMPLETED!")
    println("="^50)
end

function run_quick_test()
    d = load_dynamic_data()
    df_long = load_static_data()
    
    println("Quick Test Results:")
    println("- Static data shape: $(size(df_long))")
    println("- Dynamic data: N=$(d.N), T=$(d.T)")
    println("- State space: xbin=$(d.xbin), zbin=$(d.zbin)")
    
    θ_test = [2.0, -0.1, 1.0]
    ll = log_likelihood_dynamic(θ_test, d)
    println("- Test likelihood: $(round(ll, digits=2))")
    println("✓ Quick test completed!")
end

println("\nTest suite loaded! Available commands:")
println("  run_all_tests()  # Complete test suite")
println("  run_quick_test() # Quick functionality check")