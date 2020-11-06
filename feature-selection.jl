# This file contains everything associated with performing
# feature selection to determine which features should be
# use in a model.
#
# Copyright 2020. Rusty Conover (rusty@conover.me)
using Printf

cols_starting(prefix) = data -> filter(x -> startswith(String(x), prefix), data)
cols_containing(prefix) = data -> filter(x -> contains(String(x), prefix), data)

available_regressor_names = Dict(
    :last_demand => [:Demand_lag],

    # periodic regressors
    :daily_cycle => [:sin_288, :cos_288],
    :weekly_cycle => [:sin_2016, :cos_2016],

    # nyiso regressors.
    :nyiso_forecast_overall => [Symbol("nyiso-overall")],
    :nyiso_forecast_all => cols_starting("nyiso-"),
    :nyiso_forecast_longil => [Symbol("nyiso-longil")],
    :nyiso_forecast_mhk_valley => [Symbol("nyiso-mhk_valley")],
    :nyiso_forecast_nyc => [Symbol("nyiso-nyc")],
    :nyiso_forecast_north => [Symbol("nyiso-north")],
    :nyiso_forecast_dunwod => [Symbol("nyiso-dunwod")],
    :nyiso_forecast_west => [Symbol("nyiso-west")],
    :nyiso_forecast_centrl => [Symbol("nyiso-centrl")],
    :nyiso_forecast_capitl => [Symbol("nyiso-capitl")],
    :nyiso_forecast_genese => [Symbol("nyiso-genese")],
    :nyiso_forecast_hud_valley => [Symbol("nyiso-hud_valley")],
    :nyiso_forecast_millwd => [Symbol("nyiso-millwd")],

    # Weather regressors.
    :heat_index => cols_starting("heat_index"),
    :relative_humidity => cols_containing("relative_humidity"),
    :surface_pressure => cols_containing("surface_pressure"),
    :temperature => cols_starting("hrrr_temperature"),
    :dewpoint_temperature => cols_starting("2_metre_dewpoint"),
    :average_wind_speed => cols_containing("average_wind_speed"),
    :maximum_wind_speed => cols_containing("maximum_wind_speed"),
    :minimum_wind_speed => cols_containing("minimum_wind_speed"),
    :wind_components => cols_containing("wind_component"),
    :downward_short_wave_radiation => cols_starting("hrrr_downward"),
    :total_cloud_cover => cols_containing("total_cloud_cover"),
    :low_cloud_cover => cols_containing("low_cloud_cover"),
    :high_cloud_cover => cols_containing("high_cloud_cover"),
    :medium_cloud_cover => cols_containing("medium_cloud_cover"),
    :visible_diffuse_downward_solar_flux => cols_containing("visible_diffuse_downward"),
    :visible_beam_downward_solar_flux => cols_containing("visible_beam_downward"),
)


"""
    feature_selection_demand(epochs, trial_count, learning_rate)

Perform feature selection for electricity demand streams.

# Arguments

- `epochs`: The number of epochs to train the comparison models.
- `trial_count`: The number of trials for each model configuration
- `learning_rate`: The learning rate to use when training the
feature comparison models.

"""
function feature_selection_demand(;
    epochs::Number=100,
    trial_count::Number=1,
    learning_rate::Float64=0.001)

    electricity_load_streams = collect(filter(x -> startswith(x, "electricity-load"), keys(Microprediction.get_sponsors(Microprediction.Config()))))

    for stream_name in electricity_load_streams
        for lag_interval in [1,3,12]

            # FIXME: should only include the cities that are in
            # the particular NYISO zone, otherwise there is a lot
            # of noise.

            nyiso_feature_name = replace(replace(stream_name, "electricity-load-nyiso-" => "nyiso_forecast_"), ".json" => "")
            feature_selection(
                stream_name=stream_name,
                always_regressors=[
                    :last_demand,
                    :temperature,
                    Symbol(nyiso_feature_name),
                    ],
                never_regressors=convert(Array{Symbol,1}, []),
                forecast_locations="city",
                combination_lengths=0:2,
                filter_locations=contains(stream_name, "overall") ? false : true,
                include_nyiso=false,
                lag_interval=lag_interval,
                trial_count=trial_count,
                epochs=epochs,
            )
        end
    end
end


"""
    feature_selection_solar_power(epochs, trial_count, learning_rate)

Perform feature selection for the solar power generation streams

# Arguments

- `epochs`: The number of epochs to train the comparison models.
- `trial_count`: The number of trials for each model configuration
- `learning_rate`: The learning rate to use when training the
feature comparison models.

"""
function feature_selection_solar_power(;
    epochs::Number=100,
    trial_count::Number=1,
    learning_rate::Float64=0.001)

    for lag_interval in [1,3,12]
        feature_selection(
            stream_name="electricity-fueltype-nyiso-other_renewables.json",
            always_regressors=[
                :last_demand,
                :low_cloud_cover,
                :temperature],
            never_regressors=[
                    :daily_cycle,
                    :weekly_cycle
                ],
            forecast_locations="solar",
            combination_lengths=0:2,
            include_nyiso=false,
            lag_interval=lag_interval,
            trial_count=trial_count,
            epochs=epochs,
        )
    end

end


"""
    feature_selection_wind_power(epochs, trial_count, learning_rate)

Perform feature selection for the wind power generation streams.

# Arguments

- `epochs`: The number of epochs to train the comparison models.
- `trial_count`: The number of trials for each model configuration
- `learning_rate`: The learning rate to use when training the
feature comparison models.

"""
function feature_selection_wind_power(;
    epochs::Number=100,
    trial_count::Number=1,
    learning_rate::Float64=0.001)

    for lag_interval in [1,3,12]
        feature_selection(
            stream_name="electricity-fueltype-nyiso-wind.json",
            always_regressors=[
                :last_demand,
                :average_wind_speed,
                :wind_components,
                :relative_humidity],
            never_regressors=[
                :daily_cycle,
                :weekly_cycle,

                :downward_short_wave_radiation,
                :total_cloud_cover,
                :low_cloud_cover,
                :high_cloud_cover,
                :medium_cloud_cover,
                :visible_diffuse_downward_solar_flux,
                :visible_beam_downward_solar_flux,
            ],
            forecast_locations="wind",
            combination_lengths=0:2,
            include_nyiso=false,
            lag_interval=lag_interval,
            trial_count=trial_count,
            epochs=epochs,
        )
    end
end

"""
    feature_selection_generation(epochs, trial_count, learning_rate)

Perform feature selection for all of the generation streams.

# Arguments

- `epochs`: The number of epochs to train the comparison models.
- `trial_count`: The number of trials for each model configuration
- `learning_rate`: The learning rate to use when training the
feature comparison models.

"""
function feature_selection_generation(;
    epochs::Number=100,
    trial_count::Number=1,
    learning_rate::Float64=0.001)

    generation_streams = filter(x -> startswith(x, "electricity-fueltype"), keys(Microprediction.get_sponsors(Microprediction.Config())))

    for lag_interval in [1, 3, 12]
        for stream_name in generation_streams
            if stream_name == "electricity-fueltype-nyiso-other_renewables.json"
                feature_selection_solar_power(epochs=epochs, trial_count=trial_count, learning_rate=learning_rate)
            elseif stream_name == "electricity-fueltype-nyiso-wind.json"
                feature_selection_wind_power(epochs=epochs, trial_count=trial_count, learning_rate=learning_rate)
            else
                feature_selection(;
                    stream_name=stream_name,
                    always_regressors=[
                        :last_demand,
                        :temperature,
                    ],
                    never_regressors=convert(Array{Symbol,1}, []),
                    forecast_locations="city",
                    combination_lengths=0:2,
                    include_nyiso=false,
                    lag_interval=lag_interval,
                    trial_count=trial_count,
                    epochs=epochs,
                )
            end
        end
    end
end

function regressor_names_to_columns(regressors, data)
    cols = colnames(data)
    function handle_regressors(n)
        v = available_regressor_names[n]
        if isa(v, Array{Symbol,1})
            return Set(v)
        end
        return Set(v(cols))
    end

    return map(handle_regressors, regressors)
end

"""
    feature_selection(
        stream_name,
        forecast_locations,
        epochs,
        trial_count,
        always_regressors,
        learning_rate,
        include_nyiso,
        combination_lengths,
        lag_interval)

Perform feature selection by training multiple models and ranking
their performance by the loss function on the set of test data.

# Arguments

- `stream_name`: The stream to perform feature selection on.
- `forecast_locations`: The weather forecast locations to include
   in the features.
- `epochs`: The maximum number of epochs to train the models.
- `trial_count`: The number of individual models trained for each
configuration.  This will attempt to smooth out the randomness of
initialization.
- `always_regressors`: An array of regressors that will always be
selected for inclusion.
- `learning_rate`: The learning rate used for the model training.
- `include_nyiso`: Include the NYISO load forecasts.
- `combination_lengths`: An array of combination lengths to try.
- `lag_interval`: The forecast lag interval.

"""
function feature_selection(;
    stream_name::String,
    forecast_locations::String,
    epochs::Number=100,
    trial_count::Number=1,
    always_regressors::Array{Symbol,1},
    never_regressors::Array{Symbol,1}=[],
    filter_locations::Bool=false,
    learning_rate::Float64=0.001,
    include_nyiso::Bool=false,
    combination_lengths,
    lag_interval::Number=12)

    # The regressors that are important
    stream = loadStream(stream_name=stream_name,
                        zscore_features=true,
                        forecast_locations=forecast_locations,
                        lag_interval=lag_interval)

    optional_regressors = setdiff(keys(available_regressor_names), always_regressors)

    optional_regressors = setdiff(optional_regressors, never_regressors)

    # Filter out nyiso
    if include_nyiso == false
        optional_regressors = filter(x -> !startswith(String(x), "nyiso"), optional_regressors)
    end

    # Sub-filter the locations base on the stream name.
    if forecast_locations == "city" && filter_locations == true

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


    feature_comparison = compareFeaturePerformance(;
        stream=stream,
        always_regressor_names=always_regressors,
        optional_regressor_names=collect(optional_regressors),
        filter_regressors=filter_regressors,
        distribution=parameterizedDistribution(values(stream[1][:Demand_diff])),
        combo_lengths=combination_lengths,
        epochs=epochs,
        trial_count=trial_count,
        learning_rate=learning_rate)

    serialize("feature-selection-$(stream_name)-$(lag_interval).binary", feature_comparison)
end

struct FeatureComparisonResult
    regressors::Array{Symbol,1}
    average_loss::Float64
    minimum_loss::Float64
end

"""
    compareFeaturePerformance(
        stream,
        always_regressors,
        optional_regressors,
        combo_lengths,
        trial_count::Number=16,
        early_stopping_limit::Number=20,
        distribution,
        epochs::Number=100,
        learning_rate::Float64=0.01
    )

Compare combinations of regressors and compare their predictive
performance by comparing the loss function of the model on a
test set of data.

"""
function compareFeaturePerformance(;
    stream,
    always_regressor_names::Array{Symbol,1},
    optional_regressor_names::Array{Symbol,1},
    filter_regressors,
    combo_lengths,
    trial_count::Number=16,
    distribution,
    early_stopping_limit::Number=10,
    batchsize::Number=256,
    epochs::Number=1000,
    learning_rate::Float64=0.01)::Array{FeatureComparisonResult,1}

    # The point here is to try various combinations of regressors.
    # and figure out the most accurate model, this variable controls
    # how many different regressors will be tried at the same time.

    config_trials = []

    all_ideas = reduce(vcat, map(x -> collect(combinations(optional_regressor_names, x)), combo_lengths))

    println("Going to try $(length(all_ideas)) different regressor ideas.")

    for idea in all_ideas
        idea = sort([
            idea...,
            always_regressor_names...
            ])

        idea_column_names = regressor_names_to_columns(idea, stream[1])
        all_regressors = filter(filter_regressors, sort(collect(union(idea_column_names...))))

        println("All regressors: $(all_regressors)")
        # Training and test data need to be split, but they shouldn't be chronologially ordered.
        # because if only the recent data was used, it means that the model would be temporally
        # sensitive.

        data_columns = [all_regressors..., :Demand_diff]

        source_data = convert(Array{Float32}, values(stream[1][data_columns...]))

        source_data = shuffleobs(source_data, obsdim=1)
        train, test = splitobs(source_data, at=0.7, obsdim=1);

        # The prediction variable :Demand is at the end.
        train_x = train[:, 1:end - 1]
        train_y = train[:, end]

        test_x = test[:, 1:end - 1]
        test_y = test[:, end]

        viewize(x) = map(idx -> view(x, idx, :), 1:size(x, 1))

        regressor_list = sort(map(String, idea))

        training_loader = Flux.Data.DataLoader(viewize(train_x), train_y, batchsize=batchsize, shuffle=true)
        test_loader = Flux.Data.DataLoader(viewize(test_x), test_y, batchsize=batchsize, shuffle=true)

        model_builder = (input_count, activation, l1_regularization, l2_regularization) ->
            Chain(
                RegularizedDense(Dense(input_count, 64, activation), l1_regularization, l2_regularization),
                RegularizedDense(Dense(64, 64, activation), 0, 0),
                RegularizedDense(Dense(64, 32, activation), 0, 0),
                Dense(32, 2)
            )

        for activation in [gelu]
            model_name = "r=$(join(regressor_list, "-"))"

            println(model_name)

            trials = []
            for index in 1:trial_count
                f = @spawn trainModel(
                        model_name=model_name,
                        regressors=idea,
                        model_builder=model_builder,
                        training_loader=training_loader,
                        test_loader=test_loader,
                        activation=activation,
                        epochs=epochs,
                        distribution=distribution,
                        learning_rate=learning_rate,
                        early_stopping_limit=early_stopping_limit,
                        l1_regularization=0.05,
                        l2_regularization=0.0)
                push!(trials, f)
            end

            push!(config_trials, trials)
        end
    end

    # Satisfy all of the futures.
    println("Waiting for trial results")

    results = []
    for ct in config_trials
        trials = sort(map(fetch, ct), by=x -> x.best_test_loss)

        average_loss = mean(map(x -> x.best_test_loss, trials))
        minimum_loss = trials[1].best_test_loss
        best_model = trials[1].model

        push!(results,
            FeatureComparisonResult(
                trials[1].regressors,
                 average_loss,
                 minimum_loss))
    end

    results = sort(results, by=x -> x.average_loss)

    println("Summary:")
    println("-------------------------------")
    for record in results
        @printf "%10.3f\t%10.3f\t%s\n" record.average_loss record.minimum_loss join(map(String, record.regressors), "-")
    end
    return results
end

"""
    summarizeFeatureSelection(top_n_trials, top_n_regressors, stream_names)

Summarize the results of feature selection runs for a set of streams, by counting
the number of times each regressor appears in the top_n_trails of feature selection.

# Arguments

- `top_n_trials`: The number of top feature selection trials to consider.
- `top_n_regressors`: The number of regressors to include from the feature selection trial.
- `stream_names`: The list of stream names to consider.

"""
function summarizeFeatureSelection(
    top_n_trials::Number,
    top_n_regressors::Number,
    stream_names
    )::Dict{Number,Array{Symbol,1}}
    top_regressors_per_interval::Dict{Number,Array{Symbol,1}} = Dict()
    for lag_interval in [1, 3, 12]

        println("Lag interval: $(lag_interval)")
        regressor_usage = []

        for stream_name in stream_names
            data = deserialize("feature-selection-$(stream_name)-$(lag_interval).binary")

            for result in 1:top_n_trials
                record = data[result]
                push!(regressor_usage, record.regressors)
            end
        end

        regressor_usage = sort(collect(countmap(reduce(vcat, regressor_usage))), by=x -> x[2], rev=true)

        top_regressors_per_interval[lag_interval] = map(x -> x[1], regressor_usage[1:top_n_regressors])
    end

    return top_regressors_per_interval
end


#    demand_streams = filter(x -> startswith(x, "electricity-load"), keys(Microprediction.get_sponsors(Microprediction.Config())))
