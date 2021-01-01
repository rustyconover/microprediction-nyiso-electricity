
"""
retrieveHRRRForecast(date, forecast_offset)

# Arguments

- `forecast_date`: The date of the forecast
- `forecast_offset`: The hour offset from when the forecast was produced.

This function downlads the HRRR GRIB file from the specified date and hour,
parsed the GRIB file and serializes the returned data from parseGRIBFile()
to a filename.

"""
function retrieveHRRRForecast(forecast_date::DateTime,
                          forecast_offset::Number)

# What is the archive window of the NCEP?
    forecast_age = round(now(UTC) - forecast_date, Dates.Second)

    zero_padded_forecast_hour = lpad(Dates.Hour(forecast_date).value, 2, '0')
    zero_padded_forecast_offset = lpad(forecast_offset, 2, '0')
    fixed_date = Dates.format(forecast_date, "Ymmdd")

    if forecast_age <= Dates.Second(86400)
    # Go direct to NCEP.
        full_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.$(fixed_date)/conus/hrrr.t$(zero_padded_forecast_hour)z.wrfsfcf$(zero_padded_forecast_offset).grib2"
    else
    # At google you need to pay for transferring the forecasts out, no thanks.
    #
    # full_url = "https://storage.cloud.google.com/high-resolution-rapid-refresh/hrrr.$(fixed_date)/conus/hrrr.t$(zero_padded_forecast_hour)z.wrfsfcf$(zero_padded_forecast_offset).grib2"

        full_url = "https://pando-rgw01.chpc.utah.edu/hrrr/sfc/$(fixed_date)/hrrr.t$(zero_padded_forecast_hour)z.wrfsfcf$(zero_padded_forecast_offset).grib2"
    end

    filename = joinpath(FORECAST_DIRECTORY, "hrrr-$(fixed_date)-$(zero_padded_forecast_hour)-$(zero_padded_forecast_offset).grib2")

    if isfile(filename) == false
        res = HTTP.request(:GET, full_url, pipeline_limit=0, connection_limit=1, verbose=1, status_exception=false)

        if res.status != 200
            println("Got status $(res.status) for $(full_url) skipping")
            return
        end
        grib_download_filename, grib_out = mktemp()

        println("Opening $(grib_download_filename)")
        write(grib_out, res.body)
        close(grib_out)

        mv(grib_download_filename, filename)
    end

    parseAndSerializeGRIBFile(filename,
              fixed_date,
              zero_padded_forecast_hour,
              zero_padded_forecast_offset,
              false)
end


grib_lock = ReentrantLock()

function parseAndSerializeGRIBFile(filename, fixed_date, zero_padded_forecast_hour, zero_padded_forecast_offset, is_live)
    lock(grib_lock) do

    for (suffix, locations) in all_locations
        serialized_filename = joinpath(is_live ? LIVE_FORECAST_DIRECTORY : FORECAST_DIRECTORY,
     "$(fixed_date)-$(zero_padded_forecast_hour)-$(zero_padded_forecast_offset).forecast.$(suffix).data")

        if isfile(serialized_filename)
            continue
        end

        parse_offset_filename = joinpath(FORECAST_DIRECTORY, "grib-parse-offset.$(suffix).data")

        parse_offset_filename_temp = joinpath(FORECAST_DIRECTORY, "grib-parse-offset.$(suffix).data.temp")

        println("Parsing $(filename) for $(suffix)")
        indexes = location_to_h3_index.(locations)
        bounds = map(n -> Bounds([n]), locations)

        if !isfile(parse_offset_filename)
            parse_offsets = learnH3GribIndexes(filename, indexes, bounds, forecast_products)
            serialize(parse_offset_filename_temp, parse_offsets);

            mv(parse_offset_filename_temp, parse_offset_filename, force=true);

        end

        parse_offsets::Dict{UInt32,UInt32} = deserialize(parse_offset_filename)

        forecasts = parseGRIBFile(
            filename,
            indexes,
            parse_offsets,
            forecast_products);

        # Taking the average won't work for wind vectors.
        u_component = filter(x -> x[1]["name"] == "10 metre U wind component", forecasts)[1][2]
        v_component = filter(x -> x[1]["name"] == "10 metre V wind component", forecasts)[1][2]

        average_wind_speed = []
        minimum_wind_speed = []
        maximum_wind_speed = []

        for index in 1:length(indexes)
            wind_values = map(x -> sqrt.(sum(x.^2)), zip(u_component[index], v_component[index]))
            push!(average_wind_speed, [mean(wind_values)])
            push!(minimum_wind_speed, [minimum(wind_values)])
            push!(maximum_wind_speed, [maximum(wind_values)])
        end

        push!(forecasts, (Dict("name" => "Average Wind Speed", "level" => 10), average_wind_speed))
        push!(forecasts, (Dict("name" => "Minimum Wind Speed", "level" => 10), minimum_wind_speed))
        push!(forecasts, (Dict("name" => "Maximum Wind Speed", "level" => 10), maximum_wind_speed))


        serialize("$(serialized_filename).raw", forecasts);

        # This isn't a problem since there is just one value for the average, minimum and maximum
        # wind speeds.
        forecasts = map(x -> (x[1], mean.(x[2])), forecasts)

        serialize(serialized_filename, forecasts);
    end

    end
end


"""
retrieveForecasts()

This functions downloads and parses all weather forecasts from 2020-09-10 until
the current date.  It downloads forecasts asynchronously.

"""
function retrieveForecasts()
    start_date = DateTime(Dates.Date("2020-09-12"))
    end_date = now(UTC) + Dates.Hour(2)

    results = []
    forecast_hours = 0:1
    for forecast_date in start_date:Dates.Hour(1):end_date
        r = @spawn retrieveHRRRForecast(forecast_date, 0)
        push!(results, r)
        r = @spawn retrieveHRRRForecast(forecast_date, 1)
        push!(results, r)
    end
    map(fetch, results)

    retrieveNYISOLoadForecasts()
    return;
end


"""
    serializedHRRRForecastForTime()

Return the serialized HRRR forecast for the specified time, the actual forecast
used for the time may vary from the current hour if it has been produced or the
previous hours forecast the forecast hour incremented.

This is because NCEP's production of forecasts for the hour are sometimes
available ~47 minutes after the start of the forecast hour.

"""
function serializedHRRRForecastForTime(target::DateTime, forecast_locations::String)
    forecast_offset = 0
    current = target
    while true
        fn = liveSerializedHRRRForDateTime(current, forecast_offset)
        if fn !== nothing
            println("Loading", fn(forecast_locations))
            return deserialize(fn(forecast_locations))
        end
        # Move to the previous hour's forecast run
        current = current - Dates.Hour(1)
        # Increase the forecast offset by 1 so we're referencing the same
        # effective time.
        forecast_offset = forecast_offset + 1
    end
end

function regressor_name_for_forecast_product(product, location_name::String)::Symbol
    Symbol(lowercase("hrrr_$(replace(product["name"], " " => "_"))_$(haskey(product, "level") ? product["level"] : 0)_$(location_name)"))
end

"""
    latestHRRRForecastForTime(target, forecast_locations)

Return the the latest HRRR forecast for the specified time but perform interpolate the forecast
over the specified interpolation range of minutes.

"""

function latestHRRRForecastForTime(target::DateTime,
    forecast_locations::String;
    interpolation_range=0:1:60)



    start_time = trunc(target, Dates.Hour)
    end_time = start_time + Dates.Hour(1)
    first_values = serializedHRRRForecastForTime(start_time, forecast_locations)
    second_values = serializedHRRRForecastForTime(end_time, forecast_locations)

    first_values = fetch(first_values)
    second_values = fetch(second_values)

    field_values::Dict{Symbol,Any} = Dict()
    field_values[:datetime] = convert(Array{DateTime,1}, [])

    interpolation_range = 0:1:60

    for minutes in interpolation_range
        push!(field_values[:datetime], start_time + Dates.Minute(minutes))
    end

    location_names = get_location_name.(all_locations[forecast_locations])


    for product in [forecast_products...,
        Dict("name" => "Average Wind Speed", "level" => 10),
        Dict("name" => "Minimum Wind Speed", "level" => 10),
        Dict("name" => "Maximum Wind Speed", "level" => 10)]
        # Find that product in the serialized data.

        # Optimizing this may be intresting, because they are executed quite often.

        first_index = findfirst(x -> isSubsetOfDict(x[1], product), first_values)
        if first_index === nothing
            error("Did not find product in first hour: $(product)", println(first_values))
        end
        first_produced_product = first_values[first_index]
        second_index = first_index
        second_produced_product = second_values[second_index]

        symbol_names = Array{Symbol,1}(undef, length(first_produced_product[2]))
        for (index, value) in enumerate(zip(first_produced_product[2], second_produced_product[2]))
            field_name = regressor_name_for_forecast_product(product, location_names[index])

            if haskey(field_values, field_name) == false
                field_values[field_name] = Array{Float64,1}(undef, length(interpolation_range))
            end
            symbol_names[index] = field_name
        end

        for (first_value, second_value, field_name) in zip(first_produced_product[2], second_produced_product[2], symbol_names)
            view(field_values[field_name], 1:length(interpolation_range))[1:end] = map(function (minutes)
                first_weight = 1.0 - (minutes / 60)
                (first_weight * first_value) + ((1.0 - first_weight) * second_value)
            end, interpolation_range)
        end
    end
    return TimeArray(namedtuple(field_values), timestamp=:datetime)
end

cached_hrrr_404 = Dict()

live_hrrr_lock = ReentrantLock()
"""
    liveHRRRFilenameForDateTime(forecast_date, forecast_offset)

Return the filename in the live forecasts directory for the specified
forecast date and forecast offset, if the file does not exist it will
be tried to be downloaded.  If that fails, nothing is returned.

"""
function liveSerializedHRRRForDateTime(forecast_date::DateTime, forecast_offset::Number)
    zero_padded_forecast_hour = lpad(Dates.Hour(forecast_date).value, 2, '0')
    zero_padded_forecast_offset = lpad(forecast_offset, 2, '0')
    fixed_date = Dates.format(forecast_date, "Ymmdd")

    full_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.$(fixed_date)/conus/hrrr.t$(zero_padded_forecast_hour)z.wrfsfcf$(zero_padded_forecast_offset).grib2"

    filename = joinpath(LIVE_FORECAST_DIRECTORY, "hrrr-$(fixed_date)-$(zero_padded_forecast_hour)-$(zero_padded_forecast_offset).grib2")

    # The GRIB2 file is cached.
    if isfile(filename) == true
        return suffix -> joinpath(LIVE_FORECAST_DIRECTORY, "$(fixed_date)-$(zero_padded_forecast_hour)-$(zero_padded_forecast_offset).forecast.$(suffix).data")
    end

    if haskey(cached_hrrr_404, full_url) && cached_hrrr_404[full_url] > now()
        return nothing
    end

    lock(live_hrrr_lock) do

    res = HTTP.request(:GET, full_url; pipeline_limit=1, connection_limit=4, verbose=1, status_exception=false, readtimeout=3, retry=true)
    if res.status != 200
        # Cache the missing file for 10 minutes.
        cached_hrrr_404[full_url] = now() + Dates.Minute(10)
        return nothing
    end

    delete!(cached_hrrr_404, full_url)

    grib_download_filename, grib_out = mktemp()

    println("Opening $(grib_download_filename)")
    write(grib_out, res.body)
    close(grib_out)

    mv(grib_download_filename, filename, force=true)

    parseAndSerializeGRIBFile(filename,
        fixed_date,
        zero_padded_forecast_hour,
        zero_padded_forecast_offset,
        true)

    # Now return the
    return suffix -> joinpath(LIVE_FORECAST_DIRECTORY, "$(fixed_date)-$(zero_padded_forecast_hour)-$(zero_padded_forecast_offset).forecast.$(suffix).data")
end
end

struct SerializedForecast
    forecast_date::DateTime
    offset::Number
    filename::String
end

"""
    parseForecastFileTime(filename)

Parse out the forecast date, hour and forecast offset from the serialized
filename on the disk.

"""
function parseForecastFileTime(filename)::SerializedForecast
# Parse out the date and forecast hour.
    parts = split(filename, r"-|\.")
    forecast_date = Dates.Date(parts[1], "YYYYmmdd")
    forecast_hour = parse(Int, parts[2])
    forecast_offset = parse(Int, parts[3])

    forecast_time = DateTime(forecast_date) + Dates.Hour(forecast_hour)
    return SerializedForecast(forecast_time, forecast_offset, joinpath(FORECAST_DIRECTORY, filename))
end


cached_hrrr_forecasts = Dict()

"""
    loadHRRRForecasts()

Load all of the parsed GRIB2 forecast data and return it as a TimeArray

"""
function loadHRRRForecasts(forecast_locations)
    global cached_hrrr_forecasts
    if haskey(cached_hrrr_forecasts, forecast_locations)
        println("Old forecast cache hit")
        return cached_hrrr_forecasts[forecast_locations];
    end

    locations_descriptor = all_locations[forecast_locations]

    serialized_files = filter(m -> match(Regex("-00.forecast.$(forecast_locations).data\$"), m) !== nothing, readdir(FORECAST_DIRECTORY))

    product_names = []
    did_names = false

    field_values::Dict{Symbol,Any} = Dict()
    field_values[:datetime] = convert(Array{DateTime,1}, [])

    available_files = map(parseForecastFileTime, serialized_files)
    available_files = sort(available_files, by=x -> x.forecast_date)

    location_names = get_location_name.(locations_descriptor)

    map(x -> lowercase(replace(x[1], " " => "_")), locations_descriptor)

    for (file_index, f) in enumerate(available_files)
        summarized_forecast = deserialize(f.filename)

        # Since there may be a file for the second forecast hour, the values should be interpolated
        # between the two hours.
        second_hour = replace(f.filename, "-00.forecast" => "-01.forecast")
        next_hour_forecast = summarized_forecast
        if isfile(second_hour)
            next_hour_forecast = deserialize(second_hour)
        end

        # Now that we have two forecast results, we can interpolate the values.
        interpolation_range = 0:5:55

        for minutes in interpolation_range
            push!(field_values[:datetime], f.forecast_date + Dates.Minute(minutes))
        end


        for product in [forecast_products...,
                        Dict("name" => "Average Wind Speed", "level" => 10),
                        Dict("name" => "Minimum Wind Speed", "level" => 10),
                        Dict("name" => "Maximum Wind Speed", "level" => 10)]
            # Find that product in the serialized data.


            # Optimizing this may be intresting, because they are executed quite often.

            first_index = findfirst(x -> isSubsetOfDict(x[1], product), summarized_forecast)
            if first_index === nothing
                println("Did not find needed product in weather forecast")
                println(product)
                println("Forecast contained")
                for p in summarized_forecast
                    println(p)
                    println("$(p["name"]) $(p["level"])")
                end
                error("Did not find product in first hour: $(product)", println(summarized_forecast))
            end
            first_produced_product = summarized_forecast[first_index]

            # Optimization case:
            #
            # Right now the products are aligned between the two files so there is no need for a second
            # search to be performed since the indexes will be the same, but in the future this might
            # not be true.

            second_index = first_index
#                second_index = findfirst(x -> isSubsetOfDict(x[1], product), next_hour_forecast)
#                if length(second_index) === nothing
#                    error("Did not find product in next hour forecast: $(product)")
#                end

            second_produced_product = next_hour_forecast[second_index]

            symbol_names = Array{Symbol,1}(undef, length(first_produced_product[2]))
            for (index, value) in enumerate(zip(first_produced_product[2], second_produced_product[2]))
                field_name = regressor_name_for_forecast_product(product, location_names[index])

                if haskey(field_values, field_name) == false
                    field_values[field_name] = Array{Float64,1}(undef, length(interpolation_range) * length(available_files))
                end
                symbol_names[index] = field_name
            end

            for (first_value, second_value, field_name) in zip(first_produced_product[2], second_produced_product[2], symbol_names)
                view(field_values[field_name], (file_index - 1) * length(interpolation_range) + 1:(file_index) * length(interpolation_range))[1:end] = map(function (minutes)
                    first_weight = 1.0 - (minutes / 60)
                    (first_weight * first_value) + ((1.0 - first_weight) * second_value)
                end, interpolation_range)
            end
        end
    end

    result = TimeArray(namedtuple(field_values), timestamp=:datetime)

    cached_hrrr_forecasts[forecast_locations] = result;
    return result;
end
