using  Optim, DataFrames, CSV, HTTP, GLM,LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables
function allwrap()
cd(@__DIR__)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(minusf, startval, BFGS())
println("argmin (minimizer) is ",Optim.minimizer(result)[1])
println("min is ",Optim.minimum(result))

result_better = optimize(minusf, [-7.0], BFGS())
println(result_better)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end
"""
function ols2(beta, X, y)
    ssr = sum((y.-X*beta).^2)
    return ssr
end
"""
beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

bols = inv(X'*X)*X'*y
println("Ols closed form:",bols)
# Standard Errors
N = size(X,1)
K = size(X,2)
MSE = sum((y.-X*bols).^2)/(N-K)
var_bols = MSE*inv(X'*X)
se_bols = sqrt.(diag(var_bols))
println("Standard Errors:",se_bols)

df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::

function logit(alpha, X, d)
    pi1 = exp.(X*alpha)./(1 .+ exp.(X*alpha))
    ll = -sum((d.==1).* log.(pi1) .+ (d.==0).* log.(1 .-pi1))
    return ll
end
alpha_hat_logit = optimize(b -> logit(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
alpha_hat_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println(alpha_hat_glm)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# 5 (Dependent variable occupation)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

K = size(X,2)
J = length(unique(y))

function mlogit(alpha, X, d)
# we need to setup matrics and reshape 

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

alpha_start = randn(K * (J - 1)) # alternatively shown in the bottom
alpha_hat_optim = optimize(α -> mlogit(α, X, y),alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 1000))
alpha_hat_mle = alpha_hat_optim.minimizer
println(alpha_hat_mle)
end
allwrap()



#alternative to above way of starting value
#alpha_zero = zeros(6*size(X, 2))
#alpha_rand = rand(6*size(X, 2))
#alpha_true = [0.19259695236279964, -0.033571796904982076, 0.5967785592792664, 0.4163559566969653, -0.17476241915764645, -0.035859142540193574, 1.3070067709614859, -0.4304958272864387, 0.6890537384342846, -0.010447831660672459, 0.5232829818916963, -1.4928035717451686, -2.275673200313182, -0.005109416210297318, 1.3922965955593554, -0.9854345764973502, -1.3830137242596419, -0.014701088705079381, -0.017274436999912823, -1.493457702814482, 0.2449812454580167, -0.0067173317697710655, -0.5380537171439795, -3.789456298711893]
#alpha_start = alpha_true.*rand(size(alpha_true))
#print(size(alpha_true))
#alpha_hat_optim = optimize(α -> mlogit(α, X, y),alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 1000))
#alpha_hat_mle = alpha_hat_optim.minimizer
#println(alpha_hat_mle)
