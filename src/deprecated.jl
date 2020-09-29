Base.@deprecate_binding AbstractContext Assertion
@deprecate Context(; limit=1.0, kwargs...) Threshold(limit)
@deprecate WeightedContext(wv; limit=1.0, kwargs...) Threshold(limit; weights=wv)
