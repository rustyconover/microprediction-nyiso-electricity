# Handle all of functionality already model creation.

function build_demand_stream(;
    stream_name::String,
    save_filename_prefix::String,
    max_epochs::Number=1000,
    batch_sizes::Array{Int64,1},
    model_approach::ModelApproach,
    trial_count::Number=1)

    forecast_locations = "city"

    # Load the stream so that the symbol names of the regressors can be filtered.
    stream = loadStream(stream_name=stream_name,
        zscore_features=true,
        forecast_locations=forecast_locations,
        lag_interval=1)

    summarized_feature_selection = collect(summarizeFeatureSelection(1, 5, [stream_name]))

    for (lag_interval, regressors) in summarized_feature_selection
        println("Lag: $(lag_interval) regressors: $(regressors)")
    end
    feature_selection_preferred = Dict(
        map(x -> Pair(x[1], collect(union(regressor_names_to_columns(x[2], stream[1])...))),
        summarized_feature_selection))

    # Now filter by city if not overall.
    if forecast_locations == "city" && !contains(stream_name, "overall")
        println(stream_name)
        bad_suffixes = Set()
        for location in city_weather_stations
            if !(stream_name in location[4])
                push!(bad_suffixes, get_location_name(location))
            else
                println("Good suffix: $(get_location_name(location))")
            end
        end
        println("Bad suffixes")
        println(bad_suffixes)

        filter_regressors = function (regressor_full_name)

            # only filter regressors from the hrrr forecast
            if !startswith(String(regressor_full_name), "hrrr")
                return true
            end

            for suffix in bad_suffixes
                if endswith(String(regressor_full_name), suffix)
                    return false
                end
            end
            return true
        end
    else
        filter_regressors = (regressor_full_name) -> true
    end

    filtered_regressors::Dict{Int64,Array{Symbol,1}} = Dict()
    for (lag_interval, all_feature_names) in feature_selection_preferred
        filtered_regressors[lag_interval] = filter(filter_regressors, all_feature_names)
    end

    return buildModel(
        model_approach=model_approach,
        stream_name=stream_name,
        forecast_locations=forecast_locations,
        regressors_by_lag_interval=filtered_regressors,
        max_epochs=max_epochs,
        batch_sizes=batch_sizes,
        optimizer_names=["ADAMW"],
        trial_count=trial_count,
        save_filename_prefix=save_filename_prefix)

end

@enum ModelWindOrSolar SolarPower WindPower

"""
    build_wind_or_solar_power(save_filename_prefix, max_epochs, trial_count, lag_intervals)

Build a production wind or solar power model with the passed parameters.

# Arguments

- `save_filename_prefix`: The prefix of the saved BSON files.
- `max_epochs`: The maximum number of epochs the model will be trained for,
  if the model is no longer improving it will be stopped early.
- `trial_count`: The number of trials for each model configuration
  and architecture.  This will help solve the randomness inherent in
  initialization of the network.
- `lag_intervals`: An array of lag intervals for which the network should be trained.

"""
function build_wind_or_solar_power(;
    power_type::ModelWindOrSolar,
    save_filename_prefix::String,
    max_epochs::Number=1000,
    model_approach::ModelApproach,
    trial_count::Number=1,
    batch_size::Number=128)

    if power_type == SolarPower
        stream_name = "electricity-fueltype-nyiso-other_renewables.json"
        forecast_locations = "solar"
    else
        stream_name = "electricity-fueltype-nyiso-wind.json"
        forecast_locations = "wind"
    end

    # Load the stream so that the symbol names of the regressors can be filtered.
    stream = loadStream(stream_name=stream_name,
        zscore_features=true,
        forecast_locations=forecast_locations,
        lag_interval=1)

    summarized_feature_selection = collect(summarizeFeatureSelection(1, 5, [stream_name]))

    for (lag_interval, regressors) in summarized_feature_selection
        println("Lag: $(lag_interval) regressors: $(regressors)")
    end

    feature_selection_preferred = Dict(
        map(x -> Pair(x[1], collect(union(regressor_names_to_columns(x[2], stream[1])...))),
        summarized_feature_selection))

    return buildModel(
        stream_name=stream_name,
        model_approach=model_approach,
        forecast_locations=forecast_locations,
        regressors_by_lag_interval=feature_selection_preferred,
        learning_rates=[0.001],
        max_epochs=max_epochs,
        batch_sizes=[batch_size],
        trial_count=trial_count,
        save_filename_prefix=save_filename_prefix)
end


function makeRenewableStreams()
    @sync for model_approach in [CRPSRegression, QuantileRegression, ParameterizedDistributionDiff]
        solar = @async build_wind_or_solar_power(power_type=SolarPower,
         save_filename_prefix="t1-$(model_approach)-electricity-fueltype-nyiso-other_renewables.json",
         batch_size=256,
         max_epochs=1000,
         model_approach=model_approach)

        wind = @async build_wind_or_solar_power(power_type=WindPower,
        save_filename_prefix="t1-$(model_approach)-electricity-fueltype-nyiso-wind.json",
        batch_size=256,
        max_epochs=1000,
        model_approach=model_approach)
    end

end

function makeDemandStreams()
    all_demand_streams = [
         "electricity-load-nyiso-overall.json",
         "electricity-load-nyiso-north.json",
         "electricity-load-nyiso-nyc.json",
         "electricity-load-nyiso-mhk_valley.json",
         "electricity-load-nyiso-centrl.json",
         "electricity-load-nyiso-hud_valley.json",
         "electricity-load-nyiso-millwd.json",
         "electricity-load-nyiso-capitl.json",
         "electricity-load-nyiso-genese.json",
         "electricity-load-nyiso-west.json",
         "electricity-load-nyiso-dunwod.json",
         "electricity-load-nyiso-longil.json",
     ]

    generations = map(all_demand_streams) do stream_name
        models = map([CRPSRegression, QuantileRegression, ParameterizedDistributionDiff]) do model_approach
            @async build_demand_stream(
            stream_name=stream_name,
            model_approach=model_approach,
            save_filename_prefix="t1-$(model_approach)-$(stream_name)",
            batch_sizes=[256],
            trial_count=1,
            max_epochs=500)
        end
        return models
    end
    map(x -> map(fetch, x), generations);
end

function makeLBMPStreams()
    streams = [
         "electricity-lbmp-nyiso-north.json",
         "electricity-lbmp-nyiso-nyc.json",
          "electricity-lbmp-nyiso-mhk_valley.json",
          "electricity-lbmp-nyiso-centrl.json",
          "electricity-lbmp-nyiso-hud_valley.json",
          "electricity-lbmp-nyiso-millwd.json",
          "electricity-lbmp-nyiso-capitl.json",
          "electricity-lbmp-nyiso-genese.json",
          "electricity-lbmp-nyiso-west.json",
          "electricity-lbmp-nyiso-dunwod.json",
          "electricity-lbmp-nyiso-longil.json",
     ]
    generations = map(streams) do stream_name
        models = map([ ParameterizedDistributionDiff, CRPSRegression, QuantileRegression]) do model_approach
            @async build_lbmp_stream(
            stream_name=stream_name,
            model_approach=model_approach,
            save_filename_prefix="t1-$(model_approach)-$(stream_name)",
            batch_sizes=[256],
            trial_count=1,
            max_epochs=1000)
        end
        return models
    end
    map(x -> map(fetch, x), generations);
end


function build_lbmp_stream(;
    stream_name::String,
    save_filename_prefix::String,
    max_epochs::Number=1000,
    batch_sizes::Array{Int64,1},
    model_approach::ModelApproach,
    trial_count::Number=1)

    forecast_locations = "city"

    # Load the stream so that the symbol names of the regressors can be filtered.
    load_stream_name = replace(stream_name, "lbmp" => "load")

    stream = loadStream(stream_name=load_stream_name,
        zscore_features=true,
        forecast_locations=forecast_locations,
        lag_interval=1)

    summarized_feature_selection = collect(summarizeFeatureSelection(1, 5, [load_stream_name]))

    for lag_interval in keys(summarized_feature_selection)
        current_suffix = replace(replace(stream_name, "electricity-lbmp-nyiso-" => ""), ".json" => "")

        # Add the demand and lmbp streams for every other zone
        # but the current zone.
        for suffix in filter(x -> x !== current_suffix, nyiso_zone_suffixes)
            push!(summarized_feature_selection[lag_interval][2], Symbol("other_stream_$(suffix)"));
        end
    end


    for (lag_interval, regressors) in summarized_feature_selection
        println("Lag: $(lag_interval) regressors: $(regressors)")
    end
    feature_selection_preferred = Dict(
        map(x -> Pair(x[1], collect(union(regressor_names_to_columns(x[2], stream[1])...))),
        summarized_feature_selection))

    # Now filter by city if not overall.
    if forecast_locations == "city" && !contains(stream_name, "overall")
        println(stream_name)
        bad_suffixes = Set()
        for location in city_weather_stations
            if !(stream_name in location[4])
                push!(bad_suffixes, get_location_name(location))
            else
                println("Good suffix: $(get_location_name(location))")
            end
        end
        println("Bad suffixes")
        println(bad_suffixes)

        filter_regressors = function (regressor_full_name)

            # only filter regressors from the hrrr forecast
            if !startswith(String(regressor_full_name), "hrrr")
                return true
            end

            for suffix in bad_suffixes
                if endswith(String(regressor_full_name), suffix)
                    return false
                end
            end
            return true
        end
    else
        filter_regressors = (regressor_full_name) -> true
    end

    filtered_regressors::Dict{Int64,Array{Symbol,1}} = Dict()
    for (lag_interval, all_feature_names) in feature_selection_preferred
        filtered_regressors[lag_interval] = filter(filter_regressors, all_feature_names)
    end

    return buildModel(
        model_approach=model_approach,
        stream_name=stream_name,
        forecast_locations=forecast_locations,
        regressors_by_lag_interval=filtered_regressors,
        additional_streams=vcat(
            map(x -> "electricity-lbmp-nyiso-$(x).json", nyiso_zone_suffixes),
            map(x -> "electricity-load-nyiso-$(x).json", nyiso_zone_suffixes),
        ),
        max_epochs=max_epochs,
        batch_sizes=batch_sizes,
        optimizer_names=["ADAMW"],
        trial_count=trial_count,
        save_filename_prefix=save_filename_prefix)

end
