
"""
retrieveNYISOLoadForecasts()

Retrieve the latest load forecasts from the NYISO for all
of the different zones.

"""
function retrieveNYISOLoadForecasts()
# Load the last ten days.
    for current_date in now() - Dates.Day(9):Dates.Day(1):now() + Dates.Day(1)
        z = ZonedDateTime(current_date, tz"UTC")
        nyc_time = astimezone(z, tz"America/New_York")

        formatted_date = Dates.format(z, "Ymmdd")
        url = "http://mis.nyiso.com/public/csv/isolf/$(formatted_date)isolf.csv"

        final_filename = joinpath(NYISO_FORECAST_DIRECTORY, "$(formatted_date)isolf.csv")

        if isfile(final_filename)
            continue
        end

        res = HTTP.request(:GET, url, pipeline_limit=0, connection_limit=1, verbose=1, status_exception=false)

        if res.status != 200
            println("Got status $(res.status) for $(url) skipping")
            return
        end
        download_filename, download_out = mktemp()

        write(download_out, res.body)
        close(download_out)

        mv(download_filename, final_filename, force=true)
    end
end

"""
loadNYISOLoadForecasts()

Parse and load all of the serialized NYISO Load forecasts, return a UTC denoted
TimeArray.  Also deal with daylight savings time from the NYISO.

"""
function loadNYISOLoadForecasts()
    forecastsByZone::Dict{Symbol,Dict{DateTime,Number}} = Dict()

# All of the forecasts will exist in a directory, and they should be loaded
    forecast_files = filter(m -> match(Regex(".csv\$"), m) !== nothing, readdir(NYISO_FORECAST_DIRECTORY))
    sort!(forecast_files)

    for filename in forecast_files
        file_contents = CSV.File(joinpath(NYISO_FORECAST_DIRECTORY, filename))
        last_row = nothing
        counter = 1;

        for row in file_contents

        # Keep a count of when the last timezone appeared to handle
        # ambigious dates caused by time zones.
            if last_row == row[Symbol("Time Stamp")]
                counter = counter + 1
            else
                counter = 1
            end
            field_names = filter(x -> x != Symbol("Time Stamp"), keys(row))

        # These forecasts are eastern time not utc.

            z = ZonedDateTime(DateTime(row[Symbol("Time Stamp")], "mm/dd/YYYY HH:MM"), tz"America/New_York", counter)
            dt = DateTime(z, UTC)

            for field_name in field_names
                if !haskey(forecastsByZone, field_name)
                    forecastsByZone[field_name] = Dict()
                end

                forecastsByZone[field_name][dt] = row[field_name]
            end

            last_row = row[Symbol("Time Stamp")]
        end
    end

    # Now extract all of the seen timestamps, and build a TimeArray.
    unique_timestamps = keys(forecastsByZone[:West])

    starting_dates = sort(collect(keys(forecastsByZone[:West])))

    # Map the NYISO forecast names to the Microprediction.org stream names.
    translations::Dict{Symbol,String} = Dict(
    :Longil => "nyiso-longil",
    Symbol("Mhk Vl") => "nyiso-mhk_valley",
    Symbol("N.Y.C.") => "nyiso-nyc",
    :North => "nyiso-north",
    :Dunwod => "nyiso-dunwod",
    :West => "nyiso-west",
    :Centrl => "nyiso-centrl",
    :Capitl => "nyiso-capitl",
    :Genese => "nyiso-genese",
    :NYISO => "nyiso-overall",
    Symbol("Hud Vl") => "nyiso-hud_valley",
    :Millwd => "nyiso-millwd",
)


    interpolated_times = Array{DateTime,1}(undef, 0)
    interpolated_range = 0:1:59
    for hourly_start in starting_dates[1:end - 1]
        for minutes in interpolated_range
            push!(interpolated_times, hourly_start + Dates.Minute(minutes))
        end
    end
    push!(interpolated_times, starting_dates[end])

    v = Array{Float64,2}(undef, length(interpolated_times), length(keys(forecastsByZone)))

    forecast_zone_names = collect(keys(forecastsByZone))
    for (zone_index, zone_name) in enumerate(forecast_zone_names)
        hourly_times = map(dt -> forecastsByZone[zone_name][dt], starting_dates)

    # Now I'd like to interpolate this time array at five minute intervals.
        i = 1
        for (starting, ending) in zip(hourly_times[(1:end - 1)], hourly_times[(2:end)])
            for minutes in interpolated_range
                first_weight = 1.0 - (minutes / 60)
                second_weight = 1.0 - first_weight

                value = starting * first_weight + second_weight * ending

                v[i,zone_index] = value
                i = i + 1
            end
        end
        v[i,zone_index] = hourly_times[end]
    end
    forecast_zone_names = map(x -> Symbol(translations[x]), forecast_zone_names)

    return TimeArray(interpolated_times, v, collect(forecast_zone_names))
end
