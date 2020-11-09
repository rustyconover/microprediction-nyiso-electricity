"""
    dumpNames(filename)

Dump the available key names from a GRIB file.

"""
function dumpNames(filename)
    f = GribFile(filename)
    results = []
    for msg in f
        parsed = Message(msg)
        keylist = Vector{String}()
        for key in keys(parsed)
            push!(keylist, key)
        end
        keylist = filter(l -> match(r"(values|distinct|latitudes|longitudes|codedValues|latLonValues|bitmap)", l) === nothing, keylist)

        t =  Dict(
            map(l -> Pair(l, parsed[l]), keylist))
        push!(results, t)
#        println(results)
    end
    destroy(f)

    # Take the mean of each of those values.
    return results
end