using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions
# set the seed
Random.seed!(1234)
# Question 1 Part A
A = -5 .+ 15 .*rand(10,7)
A = rand(Uniform(-5,10), 10,7)
B = randn(10, 7) * 15 .- 2
B = rand(Normal(-2,15), 10,7)
C = [A[1:5, 1:5]  B[1:5, 6:7]]
D = ifelse.(A .<= 0, A, 0)
# Part B
length(A)

# part 2
function q2(A,B)
AB = zeros(size(A))

end
# part B
cprime = []
for r in 1:axes(c, 1)
    for c in 1: axes(c, 2)
        if c[r,c] >= -5 && c[r,c] <= 5
            push!(cprime, c[r,c])
        end
    end
end
cprime2 = c[(c.>=-5) .& (c.<=5)]

# compare the two vectors
cprime == cprime2
if cprime != cprime2
@ show size
end

# part C
x = zeros(15_169,6,5)
N = 169
# ordeing of the second dimension
# Intercept
#dummy variable
# Continuous variable normal
# normal
# binomial
# another binomial
for i in axes(x,1)
    x[i,1,:] = 1.0
    x[i,5,:] = rand(Binomial(20,0.6))
    x[i,6,:] = rand(Binomial(1,0.5))
    for t in axes(x,3)
        x[i,2,t] = rand() <= 0.75 * (6-t)/5
        x[i,3,t] = rand(Normal(15 + t -1,5*(t-1)))
        x[i,4,t] = rand(Normal(pi* (6-t), 1/exp(t)))
    end
end
# part D
β = zeros(K,T)
β[1, :] = [1+0.5*(t-1) for t in 1:T]

# part e question 2
Y = zeros(N,T)
Y = [x[:,:,t] * β[:,t] .+ rand(Normal(0,0.36),N) for t in 1:T]

return nothing
end
# part f

#3
function q3()
#part a
df = DataFrame(CSV.File("nlsw88.csv"))
@show df[1:5, :]
@show typeof(df[:, :grade])

# part B
# percentage never married
@show  mean(df[:, :never_married])
# part C
@ show mean(df[:, :grad])
# part D
@show freqtable(df[:, :race])
vars = names(df)
summarystats = describe(df)
@show summarystats

# freq table industry and occupation
@show freqtable(df[:, :industry], df[:, :occupation])

# 4 part B
function matrixops(A, B)
    out1 = A .* B
    out2 = A' * B
    out3 = sum(B+A)
    return nothing
end
# somthing written in the photo
# load("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D)
@load "matrixpractice.jld"
return nothing
end
# check the dimentions of the compatibility of the matrices
function matrixops(A::Array{Float64}, B::Array{Float64})
    if size(A) != size(B)
        throw(DimensionMismatch("Matrices A and B must have the same dimensions"))
    end
    out1 = A .* B
    out2 = A' * B
    out3 = sum(B+A)
    return  out1, out2, out3
end
matrixops(A, B)

# part f
try
    matrixops(C, D)
catch e
    println("trying matrixops(C,D):")
end

nlsw88 = DataFrame(CSV.File("nlsw88_cleaned.csv"))
ttl_exp = convert(Array, nlsw88[:, :ttl_exp])
wage = convert(Array, nlsw88.wage)])
# for unit test claude AI is used during the class with copying and pasting the code
