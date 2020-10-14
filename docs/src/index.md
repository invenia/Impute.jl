# Impute

```@setup quickstart
using DataFrames, Impute
df = Impute.dataset("test/table/neuro") |> DataFrame
```

[![stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/Impute.jl/stable/)
[![latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://invenia.github.io/Impute.jl/latest/)
[![Build Status](https://travis-ci.org/invenia/Impute.jl.svg?branch=master)](https://travis-ci.org/invenia/Impute.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/github/invenia/Impute.jl?svg=true)](https://ci.appveyor.com/project/invenia/Impute-jl)
[![codecov](https://codecov.io/gh/invenia/Impute.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/invenia/Impute.jl)

Impute.jl provides various methods for handling missing data in Vectors, Matrices and [Tables](https://github.com/JuliaData/Tables.jl).

## Installation
```julia
julia> using Pkg; Pkg.add("Impute")
```

## Quickstart

Let's start by loading our dependencies:

```@repl
using DataFrames, Impute
```

We'll also want some test data containing `missing`s to work with:

```@repl quickstart
df = Impute.dataset("test/table/neuro") |> DataFrame
```

Our first instinct might be to drop all observations, but this leaves us too few
rows to work with:

```@repl quickstart
Impute.filter(df; dims=:rows)
```

We could try imputing the values with linear interpolation, but that still leaves missing
data at the head and tail of our dataset:

```@repl quickstart
Impute.interp(df)
```

Finally, we can chain multiple simple methods together to give a complete dataset:

```@repl quickstart
Impute.interp(df) |> Impute.locf() |> Impute.nocb()
```

**Warning:**

- Your approach should depend on the properties of you data (e.g., [MCAR, MAR, MNAR](https://en.wikipedia.org/wiki/Missing_data#Types_of_missing_data)).
- In-place calls aren't guaranteedto mutate the original data, but it will try avoid copying if possible.
  In the future, it may be possible to detect whether in-place operations are permitted on an array or table using traits:
    - https://github.com/JuliaData/Tables.jl/issues/116
    - https://github.com/JuliaDiffEq/ArrayInterface.jl/issues/22
