# PS1 Julia Problem Set

## Overview
This project contains solutions and code for Problem Set 1 of the Metrics Fall 2025 course. It includes:
- Matrix operations and random data generation
- Data import and summary statistics
- Function practice and unit tests

## Requirements
- Julia 1.6 or later
- Packages: JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions, Test

## How to Run

1. Place all required data files (e.g., `nlsw88.csv`) in this directory.
2. Open Julia in this directory.
3. Run the script:
   ```julia
   include("ps1lax.jl")
   ```
4. The script will generate output files and run all unit tests.

## Running Unit Tests

Unit tests are included at the end of `ps1lax.jl` and will run automatically when you include the script.

## Notes

- Make sure to run `q1()` and `q3()` before running `q4()` to generate required files.
- If you encounter file not found errors, check your working directory and file locations.
