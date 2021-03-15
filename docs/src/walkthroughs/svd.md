# SVD Imputation

Often matrices and n-dimensional arrays with missing values can be imputed via a low rank approximation.
Impute.jl provides one such method using a single value decomposition.
The general idea is to:

1. Fill the missing values with some rough approximates (e.g., `mean`, `median`, `rand`)
2. Reconstruct this "completed" matrix with a low rank SVD approximation (i.e., `k` largest singular values)
3. Replace our initial estimates with the reconstructed values
4. Repeat steps 1-3 until convergence (update difference is below a tolerance)

To demonstrate how this is useful lets load a reduced MNIST dataset.
We'll want both the completed dataset and another dataset with 35% of the values set to `-1.0` (indicating missingness).

TODO: Update example with more a realistic dataset like some microarray data

```@example svd-example
using Distances, Impute, Plots, Statistics
mnist = Impute.dataset("test/matrix/mnist");
completed, incomplete = mnist[0.0], mnist[0.25];
```

Alright, before we get started lets have a look at what our incomplete data looks like:

```@example svd-example
heatmap(incomplete; color=:greys);
savefig("mnist-incomplete-plot.svg"); nothing # hide
```
![](mnist-incomplete-plot.svg)

Okay, so as we'd expect there's a reasonable bit of structure we can exploit.
So how does the svd method compare against other common, yet simpler, methods?

```@example svd-example
data = Impute.declaremissings(incomplete; values=-1.0)

# NOTE: SVD performance is almost identical regardless of the `init` setting.
imputors = [
    "0.5" => Impute.Replace(; values=0.5),
    "median" => Impute.Substitute(),
    "svd" => Impute.SVD(; tol=1e-2),
]

results = map(last.(imputors)) do imp
    r = Impute.impute(data, imp; dims=:)
    return nrmsd(completed, r)
end

bar(first.(imputors), results);
savefig("svd-results-plot.svg"); nothing # hide
```
![](svd-results-plot.svg)
