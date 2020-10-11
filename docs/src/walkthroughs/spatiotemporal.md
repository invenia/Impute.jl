# Spatiotemporal Panel Datasets

We often also need to missing data in spatiotemporal data.
For this example, we'll use daily temperature values from major cities around the world.

TODO: Give a different workflow/example using a DataFrame.

```@repl st-example
using AxisKeys, Impute, NamedDims, Plots, Statistics, StatsBase

# So NamedDimsArray is the outer wrapper
AxisKeys.nameouter() = true

# Construct a KeyedArray of our dataset as we want to track gaps (or missing rows)
# in the source CSV data.
data = wrapdims(
    Impute.dataset("test/table/temperature"),
    :AverageTemperature,
    :dt,
    :City;
    default=missing,
    sort=true,
)

# Rename our dims
data = rename(data, :dt => :time, :City => :loc)
```

Okay, so let's take a look at how much temperature data is missing.
```@repl st-example
heatmap(ismissing.(data); color=:greys);
savefig("st-missing-plot.svg"); nothing # hide
```
![](st-missing-plot.svg)

So many cities are missing a lot of historical data.
A common operation is to remove locations with too many missing historical observations.
In our case, we also want to penalize observations closer to the present.

Lets start be define a set of exponential weights for our observations:
```@repl st-example
wv = eweights(1:length(data.time), 0.001)
plot(wv);
savefig("st-wv-plot.svg"); nothing # hide
```
![](st-wv-plot.svg)

Now we want to filter out locations (columns) according to those weights.
For now, we'll say that a location should be removed if the weighted ratio exceeds `0.1`.
```@repl st-example
data = Impute.filter(data; dims=:cols) do v
    mratio = sum(wv[ismissing.(v)]) / sum(wv)
    return mratio < 0.1
end
```

Okay, so we removed almost 25% of the locations that didn't meet our missing data requirement.
However, most of our observations from the 1700's are still mostly missing.
Let's remove those rows that have more 50% of the locations missing.
```@repl st-example
data = Impute.filter(data; dims=:rows) do v
    mratio = count(ismissing, v) / length(v)
    return mratio < 0.5
end
```

Now let's take a look at what data remains.
```@repl st-example
heatmap(ismissing.(data); color=:greys);
savefig("st-missing-reduced-plot.svg"); nothing # hide
```
![](st-missing-reduced-plot.svg)


Alright, we can work with the remaining missing values now.
Now we could try simply imputing the values columnwise for each city using something like `Impute.nocb`
```@repl st-example
heatmap(Impute.nocb(data; dims=:cols));
savefig("st-nocb-plot.svg"); nothing # hide
```
![](st-nocb-plot.svg)

But, this looks rather crude and creates clear artifacts in the dataset.
Since we suspect that observations in similar locations would have had similar recordings
we could use `Impute.svd` or `Impute.knn` to find similarities across multiple locations.
NOTE: We need to call `svd!` on the raw data because `NamedDimsArray`s/`KeyedArray`s don't seem to support `LinearAlgebra.svd` yet.
```@repl st-example
Impute.svd!(parent(parent(data)); init=Impute.NOCB(), dims=:cols, tol=1e-2);
heatmap(data);
savefig("st-svd-plot.svg"); nothing # hide
```
![](st-svd-plot.svg)

TODO: Use KNN after fixing bug with that imputor

This method appears to have removed the artifacts found with the basic NOCB method alone.
Now we have a complete dataset ready for downstream processing :)
