# Functional

To reduce verbosity, Impute.jl also provides a functional interface to its `Validator`s, `Filter`s, `Imputor`s, etc.

Ex)

```jldoctest
julia> using Impute: Interpolate, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5];

julia> impute(M, Interpolate(); dims=:rows)
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  3.0  4.0  5.0
 1.1  2.2  3.3  4.4  5.5
```

Can also be written as
```jldoctest
julia> using Impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5];

julia> Impute.interp(M; dims=:rows)
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  3.0  4.0  5.0
 1.1  2.2  3.3  4.4  5.5
```

## Threshold
```@docs
Impute.threshold
```

## Filter
```@docs
Impute.filter
```

## Standardize
```@docs
Impute.standardize
```

## Replace
```docs
Impute.replace
```

## Substitute
```@docs
Impute.substitute
```

## Simple Random Sample (SRS)
```docs
Impute.srs
```

## Interpolate
```docs
Impute.interp
```

## Last Observation Carried Forward (LOCF)
```@docs
Impute.locf
```

## Next Observation Carried Backward (NOCB)
```@docs
Impute.nocb
```

## K-Nearest Neighbors (KNN)
```@docs
Impute.knn
```

## SVD
```@docs
Impute.svd
```
