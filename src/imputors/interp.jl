"""
    Interpolate <: Imputor

Performs linear interpolation between the nearest values in an vector.
"""
type Interpolate <: Imputor end

"""
    impute!{T<:Any}(imp::Interpolate, ctx::Context, data::AbstractArray{T, 1})

Uses linear interpolation between existing elements of a vector to fill in missing data.

WARNING: Missing values at the head or tail of the array cannot be interpolated if there
are no existing values on both sides. As a result, this method does not guarantee
that all missing values will be imputed.
"""
function impute!{T<:Any}(imp::Interpolate, ctx::Context, data::AbstractArray{T, 1})
    i = findfirst(ctx, data) + 1

    while i < length(data)
        if is_missing(ctx, data[i])
            prev_idx = i - 1
            next_idx = findnext(ctx, data, i + 1)

            if next_idx > 0
                gap_sz = (next_idx - prev_idx) - 1

                diff = data[next_idx] - data[prev_idx]
                incr = diff / T(gap_sz + 1)
                start_val = data[prev_idx]
                stop_val = data[next_idx]

                values = Real(start_val):Real(incr):Real(stop_val)

                idx_range = prev_idx:(prev_idx + length(values) - 1)
                # println(collect(idx_range))
                # println(values)

                data[idx_range] = values
                i = next_idx
            else
                break
            end
        end
        i += 1
    end

    return data
end
