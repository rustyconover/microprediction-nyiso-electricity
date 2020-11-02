# This file contains everything associated with previously
# created running saved models.
#
# Copyright 2020. Rusty Conover (rusty@conover.me)

using Microprediction

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
    save_filename_prefix::String,
    stream_update_interval::Minute=Dates.Minute(5),
    stream_name::String,
    number_of_points::Number=225,
    lag_interval::Number)

    full_filename = "$(save_filename_prefix)-lag-$(lag_interval).binary"

    saved_model = deserialize(full_filename)

    # Get the latest stream values.
    read_config = Microprediction.Config()
    live_lagged_values = Microprediction.get_lagged(read_config, stream_name)

    # Since the history doesn't name the value in the TimeArray just
    # deal with this in a hardcoded way.
    latest_value = values(live_lagged_values)[end]

    latest_stream_update_time = timestamp(live_lagged_values)[end]

    # Calculate at which the forecast should be produced
    forecast_target_time = now(UTC) + stream_update_interval * lag_interval

    live_weather = latestHRRRForecastForTime(forecast_target_time, saved_model[:forecast_locations])

    nyiso_forecast = loadNYISOLoadForecasts()

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

    # Get the start time.
    stream_start = saved_model[:stream_start]
    periodic_ticks = round((forecast_target_time - stream_start) / Dates.Millisecond(1000 * 60 * 5))

    zscore_result[:sin_288] = sin(2 * pi * periodic_ticks / 288)
    zscore_result[:cos_288] = cos(2 * pi * periodic_ticks / 288)
    zscore_result[:sin_2016] = sin(2 * pi * periodic_ticks / 2016)
    zscore_result[:cos_2016] = cos(2 * pi * periodic_ticks / 2016)

    for c in colnames(weather_data)
        # Zscore the variable by the same mu and std calculated
        # when the model was trained.
        (mu, std) = saved_model[:stats][c]
        zscore_result[c] = (values(weather_data[c])[1] - mu) / std
    end

    # Merge in all of the nyiso forecasts.
    for c in colnames(nyiso_data)
        (mu, std) = saved_model[:stats][c]
        zscore_result[c] = (values(nyiso_data[c])[1] - mu) / std
    end

    zscored_data = TimeArray(namedtuple(zscore_result); timestamp=:datetime)

    # Now build the regressor inputs for the model.
    regressor_values = convert(Array{Float32,2}, values(zscored_data[saved_model[:regressors]...]))

    model_result = saved_model[:model](vcat(regressor_values...))

    mu = model_result[1]
    std = softplus(model_result[2])
    distribution = saved_model[:distribution](mu, std)

    reverse_z(v, stat) = (v * stat[2]) + stat[1]

    # Now do the trick by iterating across the quantile function
    # thanks Peter.
    smidge = 1 / number_of_points + 2
    points = map(x -> reverse_z(quantile(distribution, x), saved_model[:stats][:Demand_diff]) + latest_value, smidge:smidge:1 - (smidge * 2))

    println("$(saved_model[:stream]) Lag=$(lag_interval) Latest value=$(latest_value) diff=(mu: $(mu) std: $(std)) points=$(mean_and_std(points)) rounded=$(mean_and_std(round.(points)))")

    return Dict(
        :stream_name => saved_model[:stream],
        :points => points)
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
#        submit_running_model(save_filename_prefix="production-solar3", stream_update_interval=Dates.Minute(5), stream_name="electricity-fueltype-nyiso-other_renewables.json")
#        println("Doing wind")
#        submit_running_model(save_filename_prefix="production-wind3", stream_update_interval=Dates.Minute(5), stream_name="electricity-fueltype-nyiso-wind.json")

#        println("Doing overall")
#        submit_running_model(save_filename_prefix="production-load-overall-no-exp", stream_update_interval=Dates.Minute(5), stream_name="electricity-load-nyiso-overall.json")

        streams = [
             "electricity-fueltype-nyiso-other_renewables.json",
             "electricity-fueltype-nyiso-wind.json",
             "electricity-load-nyiso-overall.json",
         ]
        for stream in streams
            cancelPredictions(write_key=write_key, stream_name=stream)
        end
        println("$(now()) all done")
        sleep(60 * 5)
    end
end


function submitSavedModelPrediction(;
    write_key::String,
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

    for (lag_interval, competition_delays) in lag_interval_to_competition
        output = runSavedModel(save_filename_prefix=save_filename_prefix,
                    lag_interval=lag_interval,
                    stream_update_interval=stream_update_interval,
                    stream_name=stream_name)

        points = output[:points]
        if contains(output[:stream_name], "fueltype")
            points = round.(points)
        end

        for competition_delay in competition_delays
            Microprediction.submit(write_config, output[:stream_name], convert(Array{Float64}, points), competition_delay);
        end
    end
    println("$(now()) - Finished prediction run")
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