using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions
function q1()
# set the seed
Random.seed!(1234)
# Question 1 Part a i to iv
A = -5 .+ 15 .*rand(10,7)
A = rand(Uniform(-5,10), 10,7)
B = randn(10, 7) * 15 .- 2
B = rand(Normal(-2,15), 10,7)
C = [A[1:5, 1:5]  B[1:5, 6:7]]
D = ifelse.(A .<= 0, A, 0)
# Question 1 Part b
#Number of elements of A
    println("Number of elements in A: ", length(A))
# 1 part c Number of unique elements of D
    println("Number of unique elements in D: ", length(unique(D)))
# 1. part d 'vec' operator on B using reshape
    E = reshape(B, :, 1)
    # Easier way: E2 = vec(B)
    E2 = vec(B)
# 1. Part e Make a 3-D array by stacking A and B
F = cat(A, B; dims=3)                          # 10×7×2

# 1. Part f Permute dimensions (bring slice index first)
F = permutedims(F, (3, 1, 2))

# 1. Part g Kronecker product (kron(C, F) is invalid since F is 3D)
G = kron(B, C)
println("Size of G (B ⊗ C): ", size(G))
    try
        kron(C, F)
    catch e
        println("C ⊗ F gives error: ", e)
    end

# (H) Save to JLD
save("matrixpractice.jld", "A",A, "B",B, "C",C, "D",D, "E",E, "F",F, "G",G)
# I Save first four matrices to a different JLD file
save("firstmatrix.jld",    "A",A, "B",B, "C",C, "D",D)

# part J Write C to CSV (as DataFrame for column names)
CSV.write("Cmatrix.csv", DataFrame(C, :auto))
# (k) Export D as tab-delimited .dat
    Ddf = DataFrame(D, :auto)
    CSV.write("Dmatrix.dat", Ddf; delim='\t')

    # (l) Output
    return A, B, C, D
end

# At the bottom of your script:
A, B, C, D = q1()



# Question number 2
function q2(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
# (a) Elementwise product
    AB = A .* B
# (b) Cprime with loop, Cprime2 without loop
    cprime = []
    for r in axes(C, 1)
        for s in axes(C, 2)
            if C[r, s] >= -5 && C[r, s] <= 5
                push!(cprime, C[r, s])
            end
        end
    end
cprime2 = C[(C.>=-5) .& (C.<=5)]

# compare the two vectors
cprime == cprime2
if cprime != cprime2
    @show size(cprime)
end

# part C
#Construct X: N × K × T
    N, K, T = 15_169, 6, 5
    X = zeros(N, K, T)
# ordeing of the second dimension
# Intercept
# dummy variable
# Continuous variable normal
# normal
# binomial
# another binomial
for i in 1:N
        X[i,1,:] .= 1.0
        X[i,5,:] .= rand(Binomial(20,0.6))
        X[i,6,:] .= rand(Binomial(20,0.5))
        for t in 1:T
            X[i,2,t] = rand() <= 0.75 * (6-t)/5
            X[i,3,t] = rand(Normal(15 + t -1,5*(t-1)))
            X[i,4,t] = rand(Normal(pi* (6-t)/3, 1/exp(1)))
        end
    end
# part d
β = zeros(K, T)
    β[1, :] = [1 + 0.25*(t - 1) for t in 1:T]
    β[2, :] = [log(t)            for t in 1:T]
    β[3, :] = [-sqrt(t)          for t in 1:T]
    β[4, :] = [exp(t) - exp(t+1) for t in 1:T]
    β[5, :] = [t                 for t in 1:T]
    β[6, :] = [t/3               for t in 1:T]
# part e
    σ = 0.36
    Y = zeros(N, T)
    for t in 1:T
        for i in 1:N
            Y[i, t] = dot(X[i, :, t], β[:, t]) + rand(Normal(0, σ))
        end
    end

    return X, β, Y
end
# At the bottom of script, after q1():
A, B, C, D = q1()
q2(A, B, C)

# Question 3
println("Script started")
function question3_partf(df)
    ds_sub = df[:, [:industry, :occupation, :wage]]
    grouped = groupby(ds_sub, [:industry, :occupation])
    mean_wage = combine(grouped, :wage => mean => :mean_wage)
    @show mean_wage
    return nothing
end
function q3()
#part A
df = DataFrame(CSV.File("nlsw88.csv"))
CSV.write("nlsw88_processed.csv", df)
@show df[1:5, :]
@show typeof(df[:, :grade])
# part B
# percentage never married
@show  mean(df[:, :never_married])
# part C
@show mean(df[:, :grade])
# part D
@show freqtable(df[:, :race])
vars = names(df)
summarystats = describe(df)
@show summarystats

# Part E freq table industry and occupation
@show freqtable(df[:, :industry], df[:, :occupation])
# Part (F) Mean wage over industry and occupation
question3_partf(df)
return nothing
end
println("Calling q3()")
q3()
# Question 4
"""
# matrixops(A,B) Performs the following
# (1) elementwise product of A and B,
# (2) A'B, Matrix multiplication of A transpose and B,
# (3) sum of all elements of A+B
"""
function matrixops(A, B)
# matrixops: returns (1) elementwise product, (2) A'B, (3) sum of all elements of A+B
    if size(A) != size(B)
        error("inputs must have the same size.")
    end
    out1 = A .* B
    out2 = A' * B
    out3 = sum(B+A)
    return out1, out2, out3
end
function q4()
    # (a) Load firstmatrix.jld
    @load "firstmatrix.jld" A B C D

    # (d) Evaluate matrixops with A and B
    println("matrixops(A, B):")
    out1, out2, out3 = matrixops(A, B)
    @show out1
    @show out2
    @show out3

    # (f) Evaluate matrixops with C and D (should error if sizes differ)
    println("matrixops(C, D):")
    try
        matrixops(C, D)
    catch e
        println("Error: ", e)
    end
     # (g) Evaluate matrixops with ttl_exp and wage from nlsw88_processed.csv
    df = DataFrame(CSV.File("nlsw88_processed.csv"))
    ttl_exp = convert(Array, df.ttl_exp)
    wage = convert(Array, df.wage)
    # Ensure both are column vectors of the same size
    if size(ttl_exp) != size(wage)
        ttl_exp = reshape(ttl_exp, :, 1)
        wage = reshape(wage, :, 1)
    end
    println("matrixops(ttl_exp, wage):")
    try
        out1g, out2g, out3g = matrixops(ttl_exp, wage)
        @show out1g
        @show out2g
        @show out3g
    catch e
        println("Error: ", e)
    end

    return nothing
end
# At the bottom of script we run:
q4()
# Unit tests for all functions
using Test

@testset "q1" begin
    A, B, C, D = q1()
    @test size(A) == (10, 7)
    @test size(B) == (10, 7)
    @test size(C) == (5, 7)
    @test size(D) == (10, 7)
    @test all(D[D .> 0] .== 0)
end

@testset "q2" begin
    A, B, C, D = q1()
    X, β, Y = q2(A, B, C)
    @test size(X) == (15169, 6, 5)
    @test size(β) == (6, 5)
    @test size(Y) == (15169, 5)
end

@testset "q3" begin
    q3()
    @test isfile("nlsw88_processed.csv")
end

@testset "matrixops" begin
    A = ones(3,3)
    B = fill(2.0, 3, 3)
    out1, out2, out3 = matrixops(A, B)
    @test out1 == fill(2.0, 3, 3)
    @test out2 == fill(6.0, 3, 3)
    @test out3 == sum(A+B)
    # Test error for mismatched sizes
    A2 = ones(2,2)
    @test_throws ErrorException matrixops(A, A2)
end

@testset "q4" begin
    q4()
end
# for unit test I have used AI