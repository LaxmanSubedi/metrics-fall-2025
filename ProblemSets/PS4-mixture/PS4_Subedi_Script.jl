using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions
cd(@__DIR__)
Random.seed!(1234)
include("PS4_Subedi_Source.jl")
allwrap()

#From your PS3 results, you mentioned getting:
#PS3 γ̂ = -0.094 (negative - doesn't make economic sense)
#same interpretation as PS3 , this time it's large and positive with t-stat of >100

