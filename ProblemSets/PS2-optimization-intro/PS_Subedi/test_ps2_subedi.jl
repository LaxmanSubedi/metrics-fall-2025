using Test, Random, LinearAlgebra, Distributions, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, ForwardDiff

cd(@__DIR__)

# Read in the source code
include("subedi_ps2.jl")

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 7
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# --- Question 1: Univariate optimization ---
@testset "Question 1: Univariate optimization" begin
    f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
    minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
    result = optimize(minusf, [-7.0], BFGS())
    @test isapprox(Optim.minimizer(result)[1], -7.0, atol=1.0)  # Should be near -7
end

# --- Question 2: OLS estimation ---
@testset "Question 2: OLS estimation" begin
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married .== 1

    function ols(beta, X, y)
        ssr = (y.-X*beta)'*(y.-X*beta)
        return ssr
    end

    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))
    bols = inv(X'*X)*X'*y
    @test length(beta_hat_ols.minimizer) == size(X,2)
    @test length(bols) == size(X,2)
    @test isapprox(beta_hat_ols.minimizer[2], bols[2], atol=0.1)
end

# --- Question 3: Logit estimation ---
@testset "Question 3: Logit estimation" begin
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married .== 1

    function logit(alpha, X, d)
        pi1 = exp.(X*alpha)./(1 .+ exp.(X*alpha))
        ll = -sum((d.==1).* log.(pi1) .+ (d.==0).* log.(1 .-pi1))
        return ll
    end

    alpha_hat_logit = optimize(b -> logit(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))
    @test length(alpha_hat_logit.minimizer) == size(X,2)
end

# --- Question 5: Multinomial logit estimation ---
@testset "Question 5: Multinomial logit" begin
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = dropmissing(df, :occupation)
    for occ in 8:13
        df[df.occupation .== occ, :occupation] .= 7
    end
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation
    K = size(X,2)
    J = length(unique(y))

    function mlogit(alpha, X, y)
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] .= (y .== j)
        end
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        num = zeros(N,J)
        den = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            den .+= num[:,j]
        end
        P = num ./ repeat(den, 1, J)
        loglike = -sum(bigY .* log.(P))
        return loglike
    end

    alpha_start = randn(K * (J - 1))
    alpha_hat_optim = optimize(α -> mlogit(α, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 1000))
    @test length(alpha_hat_optim.minimizer) == K * (J - 1)
end