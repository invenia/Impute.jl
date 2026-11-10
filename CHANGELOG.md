# Changelog

All notable changes to this project will be documented in this file.

## Breaking Changes

### v0.7+

- The built-in test datasets now require explicit installation of optional dependencies.
- `Impute.dataset()` and `Impute.datasets()` require `DataDeps`.
- Loading `.csv` datasets additionally requires `CSV`.
- Loading `.bson` datasets additionally requires `BSON`.

To install all dataset dependencies:

```julia
julia> using Pkg
julia> Pkg.add(["DataDeps", "CSV", "BSON"])
```

These packages are weak dependencies (Julia 1.9+) and only need to be installed if you use the test datasets.
