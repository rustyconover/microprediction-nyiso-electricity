"""
    isSubsetOfDict(a, b)

Determine if b is a subset of a if both are dictionaries.

Returns boolean
"""
function isSubsetOfDict(a, b)
    haystack = collect(a)
    needle = collect(b)
    length(intersect(haystack, needle)) === length(needle)
end

function learnH3GribIndexes(
    filename::AbstractString,
    h3_indexes::Array{UInt64},
    bounds::Array{Bounds,1},
    target_messages)

    # Build a dictionary that maps the H3 Index to an array index
    h3_to_offset::Dict{UInt64,UInt32} = Dict()
    h3_resolution = H3.API.h3GetResolution(h3_indexes[1])

    for (index, value) in enumerate(h3_indexes)
        h3_to_offset[value] = index
    end

    f = GribFile(filename)

    for msg in f
        parsed = Message(msg)

        message_def = Dict(map(n -> Pair(n, parsed[n]), GRIB_FILTERABLE_NAMES))
        # Check to see if this is a wanted message.
        # by comparing the search criteria specified via the Dict.
        found = filter(x -> isSubsetOfDict(message_def, x), target_messages)

        if length(found) == 0
            continue
        end

        # Allocate the arrays where the values will be returned.
        value_storage = Array{Float64,1}[[] for i in 1:length(h3_indexes)]

        lons, lats, values = data(msg)

        parse_offsets::Dict{UInt32,UInt32} = Dict()
        for (idx, (lon::Float64, lat::Float64, value::Float64)) in enumerate(zip(lons, lats, values))
            if !any(b -> intersects(b, lat, lon - 360.0), bounds)
                continue
            end
            h3_idx = H3.API.geoToH3(H3.Lib.GeoCoord(deg_to_rad(lat), deg_to_rad(lon)), h3_resolution)
            if haskey(h3_to_offset, h3_idx)
                parse_offsets[idx] = h3_to_offset[h3_idx]
            end
        end
        return parse_offsets
    end

    destroy(f)

    return results
end



"""
    parseGRIBFile(filename, h3_indexes, target_messages, h3_resolution)

Parse a GRIB file, extract the matching target forecast product specifications
and return the values for the specified H3 indexes.

"""
function parseGRIBFile(
    filename::AbstractString,
    h3_indexes::Array{UInt64},
    bounds::Array{Bounds,1},
    parse_offsets::Dict{UInt32,UInt32},
    target_messages)

    # Build a dictionary that maps the H3 Index to an array index
    h3_to_offset::Dict{UInt64,UInt32} = Dict()
    h3_resolution = H3.API.h3GetResolution(h3_indexes[1])

    for (index, value) in enumerate(h3_indexes)
        h3_to_offset[value] = index
    end

    f = GribFile(filename)

    results = [];

    for msg in f
        parsed = Message(msg)

        message_def = Dict(map(n -> Pair(n, parsed[n]), GRIB_FILTERABLE_NAMES))
        # Check to see if this is a wanted message.
        # by comparing the search criteria specified via the Dict.
        found = filter(x -> isSubsetOfDict(message_def, x), target_messages)

        if length(found) == 0
            continue
        end

        # Allocate the arrays where the values will be returned.
        value_storage = Array{Float64,1}[[] for i in 1:length(h3_indexes)]

        lons, lats, values = data(msg)

        for (offset, h3_offset) in parse_offsets
            push!(value_storage[h3_offset], values[offset])
        end

        for target in found
            push!(results, (message_def, value_storage))
        end
    end

    destroy(f)

    return results
end
