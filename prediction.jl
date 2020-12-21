# This file contains everything associated with previously
# created running saved models.
#
# Copyright 2020. Rusty Conover (rusty@conover.me)

using Microprediction
using Printf

cached_forecasts = Dict()
cached_nyiso_forecast = missing

cached_latest_values = Dict()

forecast_lock = ReentrantLock()

reverse_z(v, stat) = (v * stat[2]) + stat[1]


function regressorsForModel(;
    save_filename_prefix::String,
    stream_update_interval::Minute=Dates.Minute(5),
    stream_name::String,
    run_start_time::DateTime,
    number_of_points::Int64=225,
    lag_interval::Number)

    full_filename = "$(save_filename_prefix)-lag-$(lag_interval).binary"

    saved_model = deserialize(full_filename)

    if haskey(cached_latest_values, stream_name)
        live_lagged_values = cached_latest_values[stream_name]
    else
        # Get the latest stream values.
        read_config = Microprediction.Config()

        cached_latest_values[stream_name] =
        live_lagged_values = Microprediction.get_lagged(read_config, stream_name)
    end

    # Truncate it.
    live_lagged_values = to(live_lagged_values, run_start_time)

    # Since the history doesn't name the value in the TimeArray just
    # deal with this in a hardcoded way.
    latest_value = values(live_lagged_values)[end]

    latest_stream_update_time = timestamp(live_lagged_values)[end]

    # Calculate at which the forecast should be produced
    forecast_target_time = run_start_time + stream_update_interval * lag_interval

    #println("Forecast time $(forecast_target_time) last value time: $(latest_stream_update_time)")
    #println("$(forecast_target_time - latest_stream_update_time)")

    forecast_cache_key = "$(forecast_target_time)-$(saved_model[:forecast_locations])"

    live_weather = missing

    lock(forecast_lock) do
        if haskey(cached_forecasts, forecast_cache_key)
            live_weather = cached_forecasts[forecast_cache_key]
        else
            cached_forecasts[forecast_cache_key] =
            live_weather = latestHRRRForecastForTime(forecast_target_time, saved_model[:forecast_locations])
        end
    end

    global cached_nyiso_forecast
    if cached_nyiso_forecast === missing
        cached_nyiso_forecast = loadNYISOLoadForecasts()
    end
    nyiso_forecast = cached_nyiso_forecast

 #   @time nyiso_forecast = loadNYISOLoadForecasts()

    # Since the weather data is interpolated to the nearest minute
    # round appropriately
    rounded_forecast_time = round(forecast_target_time, Dates.Minute)

    # Get the weather data.
    weather_data = live_weather[rounded_forecast_time]

    nyiso_data = nyiso_forecast[rounded_forecast_time]

    zscore_result::Dict{Symbol,Any} = Dict()
    zscore_result[:datetime] = timestamp(weather_data)

    # Z-score the demand lag.
    zscore_result[:Demand_lag] = (latest_value - saved_model[:stats][:Demand_lag][1]) / saved_model[:stats][:Demand_lag][2]

    println("Last demand", latest_value);

    # Get the start time.
    stream_start = saved_model[:stream_start]
    periodic_ticks = round((forecast_target_time - stream_start) / Dates.Millisecond(1000 * 60 * 5))

    println("Periodic tics", periodic_ticks)
    zscore_result[:sin_288] = sin(2 * pi * periodic_ticks / 288)
    zscore_result[:cos_288] = cos(2 * pi * periodic_ticks / 288)
    zscore_result[:sin_2016] = sin(2 * pi * periodic_ticks / 2016)
    zscore_result[:cos_2016] = cos(2 * pi * periodic_ticks / 2016)

    for c in colnames(weather_data)
        # Zscore the variable by the same mu and std calculated
        # when the model was trained.
        (mu, std) = saved_model[:stats][c]
        v = values(weather_data[c])[1]
        if contains(String(c), "elmira") && contains(String(c), "temperature")
        println("$(c) == $(v)")
        end
        zscore_result[c] = (v - mu) / std
    end

    # Merge in all of the nyiso forecasts.
    for c in colnames(nyiso_data)
        (mu, std) = saved_model[:stats][c]
        zscore_result[c] = (values(nyiso_data[c])[1] - mu) / std
    end

    zscored_data = TimeArray(namedtuple(zscore_result); timestamp=:datetime)

    # Now build the regressor inputs for the model.
    regressor_values = convert(Array{Float32,2}, values(zscored_data[saved_model[:regressors]...]))

    return vcat(regressor_values...), latest_value
end

"""
    runSavedModel(
        safe_filename_prefix::String,
        stream_update_interval::Minute=Dates.Minute(5),
        stream_name::String,
        number_of_points::Number=225,
        lag_interval::Number
    )

Execute a saved model to produce a set of points that describe
the predicted distribution.

"""
function runSavedModel(;
    identity_name::String,
    save_filename_prefix::String,
    stream_update_interval::Minute=Dates.Minute(5),
    stream_name::String,
    run_start_time::DateTime,
    number_of_points::Int64=225,
    all_candidates::Bool=false,
    lag_interval::Number)

    full_filename = "$(save_filename_prefix)-lag-$(lag_interval).binary"

    saved_model = deserialize(full_filename)


    if all_candidates === true
        models = map(x -> (x[:training_result].model, x[:training_result].name), saved_model[:all_results])
    else
        models = [(saved_model[:model], saved_model[:model_name])]
    end

    regressors, latest_value = regressorsForModel(
        save_filename_prefix=save_filename_prefix,
        stream_update_interval=stream_update_interval,
        stream_name=stream_name,
        run_start_time=run_start_time,
        number_of_points=number_of_points,
        lag_interval=lag_interval
    )
    results = []
    for (model, name) in models

        # Lets have some fun and run the model in javascript.

        if saved_model[:model_approach] !== CRPSRegressionSeperate
            # All of these model approaches just build a single result
            model_result = model(regressors)
        else
            # This approach uses two models.
            base_result = model[1](regressors)
            distribution_result = model[2](regressors)
        end

        if saved_model[:model_approach] === ParameterizedDistributionDiff
            mu = model_result[1]
            std = softplus(model_result[2])

            distribution = saved_model[:distribution](mu, std)

            # Now do the trick by iterating across the quantile function
            # thanks Peter.
            smidge = 1 / (number_of_points + 2)
            points = map(x -> reverse_z(quantile(distribution, x), saved_model[:stats][:Demand_diff]) + latest_value, smidge:smidge:1 - (smidge * 2))
        elseif saved_model[:model_approach] === ClosestPoint
            first_model_result = view(cumsum(vcat(
                view(model_result, 1:1, :),
                view(model_result, 2:size(model_result, 1), :).^2
            ), dims=1), 2:size(model_result, 1), :)

            points = map(x -> reverse_z(x, saved_model[:stats][:Demand]), first_model_result)
        elseif saved_model[:model_approach] === CRPSRegressionSeperate
            first_model_result = view(cumsum(vcat(
                base_result,
                distribution_result,
            ), dims=1), 2:226, :)

            points = map(x -> reverse_z(x, saved_model[:stats][:Demand]), first_model_result)
        elseif saved_model[:model_approach] === QuantileRegression || saved_model[:model_approach] === CRPSRegression
            first_model_result = view(cumsum(vcat(
                view(model_result, 1:1, :),
                view(model_result, 2:size(model_result, 1), :).^2
            ), dims=1), 2:size(model_result, 1), :)

            points = map(x -> reverse_z(x, saved_model[:stats][:Demand]), first_model_result)
        end

        points_stats = mean_and_std(points)
        output = @sprintf "%s %s %s %s lag=%d latest=%f points mu=%.2f std=%.2f" name run_start_time identity_name saved_model[:stream] lag_interval latest_value points_stats[1] points_stats[2]
        println(output)

        push!(results, Dict(
            :stream_name => saved_model[:stream],
            :points => points,
            :model_result => model_result,
            :name => name
        ))

    end

    if all_candidates === false
        return results[1]
    end

    return results
end


"""
    runSavedModels(write_key)

Execute all of the saved models and send their predictions to
Microprediction.

"""
function runSavedModels(write_key::String="8f0fb3ce57cb67498e3790f9d64dd478")
    while true
#        println("$(now()) Starting prediction run")
#        println("Doing solar")

        # Set the time for ths model run so the forecasts can
        # hopefully be cached.
        run_time = now(UTC)
        global cached_forecasts = Dict()
        global cached_latest_values = Dict()

        # Non points stream.
        write_key =
        all_demand_streams = [
            "electricity-load-nyiso-overall.json",
            "electricity-load-nyiso-north.json",
            "electricity-load-nyiso-centrl.json",
            "electricity-load-nyiso-hud_valley.json",
            "electricity-load-nyiso-millwd.json",
            "electricity-load-nyiso-mhk_valley.json",
            "electricity-load-nyiso-nyc.json",
            "electricity-load-nyiso-capitl.json",
            "electricity-load-nyiso-genese.json",
            "electricity-load-nyiso-west.json",
            "electricity-load-nyiso-dunwod.json",
            "electricity-load-nyiso-longil.json",
        ]

        submissions = []
        map(all_demand_streams) do stream_name
            m = @async submitSavedModelPrediction(
                run_start_time=run_time,
                identity_name="Flex Hedgehog",
                write_key="8f0fb3ce57cb67498e3790f9d64dd478", # Flex Hedgehog
                save_filename_prefix="demand-$(stream_name)",
                stream_update_interval=Dates.Minute(5),
                stream_name=stream_name)
            push!(submissions, m)

            # points based models
            m = @async submitSavedModelPrediction(
                    run_start_time=run_time,
                    identity_name="Message Moose",
                    write_key="7e5d0f66b23def57c5f9bcee73ab45dd", # Message Moose
                    save_filename_prefix="demand-points-$(stream_name)",
                    stream_update_interval=Dates.Minute(5),
                    stream_name=stream_name)
            push!(submissions, m)

        end

        #
        for stream_name in ["electricity-load-nyiso-overall.json"]
            m = @async submitSavedModelPrediction(
                run_start_time=run_time,
                identity_name="Ghetto Beetle",
                write_key="6330e620ad685214a3ace39e67108696",  # Stealthy Flea
                save_filename_prefix="demand-qr-point-5-$(stream_name)",
                stream_update_interval=Dates.Minute(5),
                stream_name=stream_name)
            push!(submissions, m)
        end

        map(fetch, submissions)

        finish_time = now(UTC)

        println("Time taken to forecast $(finish_time - run_time)")

        sleep_time = 60 * 5 * 1000 - (finish_time - run_time).value;
        println("Sleeping for $(sleep_time/1000.0) seconds")
        sleep(sleep_time/1000.0)
    end
end

logging_lock = ReentrantLock()

function submitSavedModelPrediction(;
    write_key::String,
    identity_name::String,
    run_start_time::DateTime,
    save_filename_prefix::String,
    stream_update_interval::Minute,
    stream_name::String)

    lag_interval_to_competition = Dict(
        1 => [70, 310],
        3 => [910],
        12 => [3555]
    )

    # Send the prediction in.
    write_config = Microprediction.Config(write_key)

    for (lag_interval, competition_delays) in sort(collect(lag_interval_to_competition))
        output = runSavedModel(
            save_filename_prefix=save_filename_prefix,
            identity_name=identity_name,
            lag_interval=lag_interval,
            run_start_time=run_start_time,
            stream_update_interval=stream_update_interval,
            stream_name=stream_name)

        points = output[:points]
        if contains(output[:stream_name], "fueltype")
            points = round.(points)
        end

        # Should write this to a file.

        lock(logging_lock) do
            io = open("prediction-log.json", "a")
            for competition_delay in competition_delays
                println(io,
                JSON.json(Dict(
                    :stream_name => stream_name,
                    :write_key => write_key,
                    :prediction_run_time => run_start_time,
                    :time => convert(Int64, round(datetime2unix(now(UTC)))),
                    :points => points,
                    :save_filename_prefix => save_filename_prefix,
                    :delay => competition_delay)))
                Microprediction.submit(write_config, output[:stream_name], convert(Array{Float64}, points), competition_delay);
            end
            close(io)
        end
    end

end

function cancelPredictions(;
    write_key::String,
    stream_name::String)
    # Send the prediction in.
    write_config = Microprediction.Config(write_key)

    lag_interval_to_competition = Dict(
        1 => [70, 310],
        3 => [910],
        12 => [3555]
    )

    for (lag_interval, competition_delays) in lag_interval_to_competition
        for competition_delay in competition_delays
            @async Microprediction.cancel(write_config, stream_name, competition_delay)
        end
    end
end#