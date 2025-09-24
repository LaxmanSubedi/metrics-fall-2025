using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables
# Set working directory and run analysis
cd(@__DIR__)
include("PS3_subedi_code.jl")
allwrap()

#---------------------------------------------------
# Results Interpretation
#---------------------------------------------------
# estimated parameters for multinomial logit
# estimated gamma is -0.09418795907511646
# gamma represents the the change in latent utility
# with 1 unit change in the relative E(log wage)
# in occupation j (relative to other)
# positive gamma is intuitive as people liked earning more
# negative gamma is surprising as people liked earning more
# probabily encountered mis-specification

