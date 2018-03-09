# Impute
[![stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/Impute.jl/stable/)
[![latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://invenia.github.io/Impute.jl/latest/)
[![Build Status](https://travis-ci.org/invenia/Impute.jl.svg?branch=master)](https://travis-ci.org/invenia/Impute.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/github/invenia/Impute.jl?svg=true)](https://ci.appveyor.com/project/invenia/Impute-jl)
[![codecov](https://codecov.io/gh/invenia/Impute.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/invenia/Impute.jl)

Impute.jl provides various data imputation methods for `Arrays` and `DataFrames` with various types of missing values.

## Installation
```julia
Pkg.clone("https://github.com/invenia/Impute.jl")
```

## Features
* Operate over Vectors, Matrices or DataFrames
* Chaining of methods

## Methods

* drop - remove missing
* locf - last observation carried forward
* nocb - next observation carried backward
* interp - linear interpolation of values in vector
* fill - replace with a specific value or a function which returns a value given the existing vector with missing values dropped.

## TODO

* Dropping rows in a matrix allocates extra memory (ie: `data[mask, :]` make a copy).
* More sophisticated imputation methods
    1. MICE
    2. EM
    3. kNN
    4. Regression
