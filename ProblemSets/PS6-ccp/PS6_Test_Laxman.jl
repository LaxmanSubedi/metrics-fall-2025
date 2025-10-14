using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

cd(@__DIR__)

include("PS6_Source_Laxman.jl")

# comprehensive Unit Test suite for PS6_Source_Laxman.jl
@testset "PS6 Rust Model CCP Estimation Tests" begin
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Q1: Data Loading and Reshaping
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Q1: Data Loading" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
        df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
        
        # Type and dimension tests
        @test isa(df_long, DataFrame) && isa(Xstate, Matrix)
        @test size(Xstate, 2) == 20
        @test length(Zstate) == length(Branded) == size(Xstate, 1)
        @test nrow(df_long) == size(Xstate, 1) * 20
        
        # Required columns
        @test all([:bus_id, :time, :Y, :Odometer, :Xstate, :RouteUsage, :Branded, :Zst] .∈ Ref(names(df_long)))
        
        # Data validity
        @test all(0 .<= df_long.Y .<= 1)
        @test all(df_long.Xstate .>= 1)
        @test all(df_long.time .>= 1) && all(df_long.time .<= 20)
        @test issorted(df_long, [:bus_id, :time])
        @test !any(ismissing, df_long.Y)
        
        println("✓ Data tests passed - $(size(Xstate,1)) buses, $(nrow(df_long)) obs")
    end
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Q2: Flexible Logit
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Q2: Flexible Logit" begin
        Random.seed!(123)
        test_df = DataFrame(
            Y = rand(0:1, 100),
            Odometer = rand(50000:300000, 100),
            RouteUsage = rand([0.1, 0.5, 0.9], 100),
            Branded = rand(0:1, 100),
            time = rand(1:20, 100)
        )
        
        flex_model = estimate_flexible_logit(test_df)
        
        @test isa(flex_model, GeneralizedLinearModel)
        @test GLM.family(flex_model) == Binomial()
        @test all(isfinite, coef(flex_model))
        @test GLM.converged(flex_model)
        
        preds = predict(flex_model, test_df)
        @test all(0 .<= preds .<= 1)
        @test length(preds) == 100
        
        println("✓ Flexible logit tests passed")
    end
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Q3a: State Space Construction
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Q3a: State Space" begin
        xbin, zbin = 3, 2
        xval, zval = [100.0, 150.0, 200.0], [0.2, 0.8]
        xtran = rand(6, 3)
        
        state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
        
        @test nrow(state_df) == xbin * zbin
        @test ncol(state_df) == 4
        @test all([:Odometer, :RouteUsage, :Branded, :time] .∈ Ref(names(state_df)))
        @test all(state_df.Odometer .∈ Ref(xval))
        @test all(state_df.RouteUsage .∈ Ref(zval))
        @test all(state_df.Branded .== 0)
        @test all(state_df.time .== 0)
        
        # Kronecker structure
        @test state_df.Odometer == kron(ones(zbin), xval)
        @test state_df.RouteUsage == kron(zval, ones(xbin))
        
        println("✓ State space tests passed")
    end
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Q3b: Future Values
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Q3b: Future Values" begin
        Random.seed!(456)
        xbin, zbin, T, β = 3, 2, 5, 0.9
        xval, zval = [100.0, 150.0, 200.0], [0.3, 0.7]
        xtran = rand(6, 3)
        
        state_df = DataFrame(
            Odometer = kron(ones(zbin), xval),
            RouteUsage = kron(zval, ones(xbin)),
            Branded = zeros(6),
            time = zeros(6)
        )
        
        train_data = DataFrame(
            Y = rand(0:1, 50),
            Odometer = rand(xval, 50),
            RouteUsage = rand(zval, 50),
            Branded = rand(0:1, 50),
            time = rand(1:T, 50)
        )
        
        mock_model = glm(@formula(Y ~ Odometer + RouteUsage + Branded + time), 
                        train_data, Binomial(), LogitLink())
        
        FV = compute_future_values(state_df, mock_model, xtran, xbin, zbin, T, β)
        
        @test size(FV) == (6, 2, T + 1)
        @test all(isfinite, FV)
        @test all(FV[:, :, 1] .== 0)    # Initial period
        @test all(FV[:, :, T+1] .== 0)  # Terminal period
        @test all(FV[:, :, 2:T] .<= 0)  # Negative values
        
        # Test β = 0 (no discounting)
        FV_zero = compute_future_values(state_df, mock_model, xtran, xbin, zbin, T, 0.0)
        @test all(FV_zero .== 0)
        
        println("✓ Future values tests passed")
    end
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Q3c: FVT1 Mapping
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Q3c: FVT1 Mapping" begin
        Random.seed!(789)
        N, T, xbin, zbin = 4, 5, 3, 2
        
        df_test = DataFrame(
            bus_id = repeat(1:N, inner=T),
            time = repeat(1:T, outer=N),
            Y = rand(0:1, N*T),
            Odometer = rand(100000:250000, N*T),
            Xstate = rand(1:xbin, N*T),
            RouteUsage = rand([0.3, 0.7], N*T),
            Branded = repeat(rand(0:1, N), inner=T),
            Zst = repeat(rand(1:zbin, N), inner=T)
        )
        
        Xstate_test = rand(1:xbin, N, T)
        Zstate_test = rand(1:zbin, N)
        Branded_test = rand(0:1, N)
        
        FV = rand(xbin * zbin, 2, T + 1) .* (-1)
        FV[:, :, 1] .= 0
        FV[:, :, T+1] .= 0
        xtran = rand(xbin * zbin, xbin)
        
        fvt1_result = compute_fvt1(df_test, FV, xtran, Xstate_test, Zstate_test, xbin, Branded_test)
        
        @test length(fvt1_result) == N * T
        @test all(isfinite, fvt1_result)
        @test isa(fvt1_result, Vector{Float64})
        
        # Can reshape to matrix
        FVT1_matrix = reshape(fvt1_result, T, N)'
        @test size(FVT1_matrix) == (N, T)
        
        println("✓ FVT1 mapping tests passed")
    end
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Q3d: Structural Parameters
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Q3d: Structural Params" begin
        Random.seed!(321)
        n = 150
        
        struct_df = DataFrame(
            Y = rand(0:1, n),
            Odometer = rand(100000:300000, n),
            Branded = rand(0:1, n)
        )
        
        fvt1 = randn(n) .* 0.1
        
        theta_model = estimate_structural_params(struct_df, fvt1)
        
        @test isa(theta_model, GeneralizedLinearModel)
        @test GLM.family(theta_model) == Binomial()
        @test all(isfinite, coef(theta_model))
        @test GLM.converged(theta_model)
        @test length(coef(theta_model)) >= 2
        
        preds = predict(theta_model)
        @test all(0 .<= preds .<= 1)
        @test length(preds) == n
        
        println("✓ Structural parameter tests passed")
    end
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Integration & Edge Cases
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Integration & Robustness" begin
        # Main function runs
        @test_nowarn main()
        
        # Minimal state space
        min_state = construct_state_space(2, 2, [100.0, 200.0], [0.3, 0.7], rand(4, 2))
        @test nrow(min_state) == 4
        
        # Function existence
        @test all([isdefined(Main, f) for f in [
            :load_and_reshape_data, :estimate_flexible_logit, 
            :construct_state_space, :compute_future_values, 
            :compute_fvt1, :estimate_structural_params, :main
        ]])
        
        # Method signatures
        @test hasmethod(load_and_reshape_data, (String,))
        @test hasmethod(estimate_flexible_logit, (DataFrame,))
        @test hasmethod(construct_state_space, (Int, Int, Vector, Vector, Matrix))
        @test hasmethod(estimate_structural_params, (DataFrame, Vector))
        @test hasmethod(main, ())
        
        println("✓ Integration tests passed")
    end
    
end

println("\n" * "="^60)
println("PS6 RUST MODEL TEST SUITE COMPLETED")
println("All $(Test.get_testset().n_passed) tests passed successfully!")
println("="^60)
