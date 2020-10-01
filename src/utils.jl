function trycopy(data)
    # Not all objects support `copy`, but we should use it to improve
    # performance if possible.
    try
        copy(data)
    catch
        deepcopy(data)
    end
end
