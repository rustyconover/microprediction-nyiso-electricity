# Build a framework for the prediction of NYISO streams.
#
# Author: Rusty Conover <rusty@conover.me>
#
# Description:
#
# The ideas that I've had for forecasting the various NYISO
# electricty streams are:
#
# 1. Combine weather forecast products from the HRRR produced by
#    NCEP to better forecast renewable generation.
#   a. Wind: extract a very local forecast of the winds for each
#      wind farm as shown in the Gold Book
#   b. Locate all of the solar generation facilities, and extract
#      a localized forecast including the cloud cover and the solar
#      radiation to attempt to predict generation.
#
# 2. Replicate the weather locations from NYISO's modeling process
#    which are based on population around the state for the various
#    generation zones to estimate demand.
#
#    The day ahead forecasting model is explained at:
#    https://www.nyiso.com/documents/20142/2923301/dayahd_schd_mnl.pdf/0024bc71-4dd9-fa80-a816-f9f3e26ea53a
#
#    The weather variables used are:
#       Temperature, dew point, cloud cover, wind speed.
#
# 3. Obtain the average price of natural gas from various pipelines
#    serving New York to determine if the generation will prefer
#    natural gas vs coal or heating oil.  (still pending dataset)
#
# 4. NYISO prepares a day ahead hourly load forecast for each zone
#    it seems that combining the value of the forecast in the model
#    is likely to be productive as it factors in holidays and has
#    a longer training set.

using Combinatorics
using Dates
using DistributionsAD
using Flux
using GRIB
using H3
using HTTP
using LinearAlgebra
using MicropredictionHistory
using NamedTupleTools
using OhMyREPL
using Plots
using Serialization
using StatsBase
# using TensorBoardLogger
using TimeSeries
using CSV
using TimeZones
using MLDataUtils
using BSON
using Impute
using JSON
using Statistics
using RollingFunctions
using KernelDensity

@everywhere BLAS.set_num_threads(1)

# Locations that have wind generation.
wind_generation_sites = [
    ("Altona", 44.891389, -73.655833),
    ("Arkwright", 42.389722, -79.243056),
    ("Avoca", 42.410278, -77.425556),
    ("Bliss", 42.548611, -78.298056),
    ("Bouckville", 42.900833, -75.515278),
    ("Chateaugay", 44.9225, -74.074167),
#    ("Clinton", 44.943056, -73.9475), Same as Chateaugay
    ("Copenhagen", 43.893056, -75.6725),
    ("Ellenburg", 44.893889, -73.836667),
    ("Fairfield", 43.124722, -74.934444),
    ("Fenner", 42.978333, -75.781111),
#    ("Howard", 42.364167, -77.508889), Same as Avoca
    ("Jasper", 42.1225, -77.503056),
    ("Lackawanna", 42.819444, -78.825556),
    ("Lowville", 43.786667, -75.492222),
    ("Orangeville", 42.7375, -78.265278),
    ("Sheldon", 42.737222, -78.418611),
    ("Wethersfield", 42.651667, -78.235556)
];

# Solar generation sites.
solar_generation_sites = [
    ("Upton", 40.869444, -72.886667),
    ("East Shoreham", 40.950278, -72.885278),
]

# City based weather stations.
city_weather_stations = [
    ("Albany", 42.6525, -73.757222, ["electricity-load-nyiso-capitl.json"]),
    ("Binghampton", 42.102222, -75.911667, ["electricity-load-nyiso-centrl.json"]),
    ("Buffalo", 42.904722, -78.849444, ["electricity-load-nyiso-west.json"]),
    ("Canandaigua", 42.8875, -77.281667, ["electricity-load-nyiso-genese.json"]),
    ("Elmira", 42.093889, -76.809722, ["electricity-load-nyiso-centrl.json"]),
    ("White Plains", 41.04, -73.778611, ["electricity-load-nyiso-millwd.json", "electricity-load-nyiso-dunwod.json"]),
    ("Islip", 40.756667, -73.198889, ["electricity-load-nyiso-longil.json"]),
    ("JFK", 40.639722, -73.778889, ["electricity-load-nyiso-nyc.json"]),
    ("Jamestown", 42.095556, -79.238611, ["electricity-load-nyiso-west.json"]),
    ("Fredonia", 42.440833, -79.333889, ["electricity-load-nyiso-west.json"]),
    ("NYC", 40.748817, -73.985428, ["electricity-load-nyiso-nyc.json"]),
    ("LGA", 40.77725, -73.872611, ["electricity-load-nyiso-nyc.json"]),
    ("Staten Island", 40.571944, -74.146944, ["electricity-load-nyiso-nyc.json"]),
    ("Massena", 44.930278, -74.8925, ["electricity-load-nyiso-mhk_valley.json"]),
    ("Monticello", 41.653611, -74.690556, ["electricity-load-nyiso-mhk_valley.json"]),
    ("Plattsburgth", 44.695278, -73.458333, ["electricity-load-nyiso-north.json"]),
    ("Poughkeepsie", 41.7, -73.93, ["electricity-load-nyiso-hud_valley.json"]),
    ("Rochester", 43.165556, -77.611389, ["electricity-load-nyiso-genese.json"]),
    ("Newburgh", 41.519722, -74.021389, ["electricity-load-nyiso-hud_valley.json"]),
    ("Schenectady", 42.814167, -73.937222, ["electricity-load-nyiso-capitl.json"]),
    ("Syracuse", 43.046944, -76.144444, ["electricity-load-nyiso-centrl.json"]),
    ("Rome", 43.219444, -75.463333, ["electricity-load-nyiso-mhk_valley.json"]),
    ("Utica", 43.0970142,  -75.2279416, ["electricity-load-nyiso-mhk_valley.json"]),
    ("Watertown", 43.975556, -75.906389, ["electricity-load-nyiso-mhk_valley.json"]),
];


# This is the resolution of the H3 indexes to use when aggregating weather
# forecast information.
#
# https://github.com/uber/h3/blob/master/docs/core-library/restable.md
##
SITE_H3_RESOLUTION = 6

# H3 likes to use radians rather than degrees for its coordinates, so provide some conversions.
deg_to_rad(x) = x * pi / 180
rad_to_deg(x) = x * 180 / pi
location_to_h3_index(x) = H3.API.geoToH3(H3.Lib.GeoCoord(deg_to_rad(x[2]), deg_to_rad(x[3])), SITE_H3_RESOLUTION)
get_location_name(x) = lowercase(replace(x[1], " " => "_"))

# Check for duplicates
function check_duplicates()
    seen_locations = Set()
    for location in city_weather_stations
        l = location_to_h3_index(location)
        if l in seen_locations
            println("Duplicate", location)
        end
        push!(seen_locations, l)
    end
end

# Store all of the location collections. so they can be easily referenced later on as needed
all_locations = Dict(
    "solar" => solar_generation_sites,
    "city" => city_weather_stations,
    "wind" => wind_generation_sites
);


# To keep the number of filterable names short for comparisons,
# only compare these values from GRIB files.
GRIB_FILTERABLE_NAMES = ["name", "shortName", "level", "typeOfLevel", "units"]

# These are the interesting forecast products for the model, they are matched
# against the produced forecast layers, to try new features in your own models
# add additional products here.
forecast_products = [
    Dict("name" => "Temperature", "level" => 0),
    Dict("name" => "Surface pressure"),
    Dict("name" => "2 metre dewpoint temperature", "level" => 2),
    Dict("name" => "2 metre relative humidity", "level" => 2),
    Dict("name" => "10 metre U wind component", "level" => 10),
    Dict("name" => "10 metre V wind component", "level" => 10),
    Dict("name" => "Downward short-wave radiation flux"),
    Dict("name" => "Visible Beam Downward Solar Flux", "level" => 0),
    Dict("name" => "Visible Diffuse Downward Solar Flux", "level" => 0),
    Dict("name" => "Total Cloud Cover", "level" => 0),
    Dict("name" => "Low cloud cover", "level" => 0),
    Dict("name" => "Medium cloud cover", "level" => 0),
    Dict("name" => "High cloud cover", "level" => 0),
]

const quantile_count = 225
const quantile_increment = convert(Float32, 0.995 / quantile_count)
const fixed_quantiles = convert(Array{Float32,1}, map(x -> quantile_increment * x, 1:quantile_count))
const inverse_fixed_quantiles = convert(Array{Float32,1}, fixed_quantiles .- 1.0)
const big_quantiles = hcat(fixed_quantiles, inverse_fixed_quantiles)


# Where should the forecast files be stored.
FORECAST_DIRECTORY = "/Users/rusty/Data/weather-forecasts/"

# The directory where the Microprediction.org stream history is stored.
MICROPREDICTION_HISTORY_DIRECTORY = "/Users/rusty/Development/pluto/data"

# The date at which Microprediction.org history became reliable.
MICROPREDICTION_HISTORY_START_DATE = DateTime(2020, 9, 15, 0, 0, 0)

# The directory where NYISO forecasts should be stored.
NYISO_FORECAST_DIRECTORY = "/Users/rusty/Data/nyiso-load/"


LIVE_FORECAST_DIRECTORY = "/Users/rusty/Data/live-weather-forecasts/"

@enum ModelApproach ParameterizedDistributionDiff CRPSRegression QuantileRegression ClosestPoint CRPSRegressionSeperate

include("bounds.jl")
include("grib.jl")

include("nyiso-forecast.jl")
include("hrrr.jl")

include("regularized-dense-layer.jl")

include("prediction.jl")
include("feature-selection.jl")
include("plots.jl")

include("model-creation.jl")
include("loss-functions.jl")
include("serialize-to-js.jl")

"""
    loadStream(forecast_locations,
               zscore_features::Bool=true,
               need_full_history::Bool=true,
               load_live_data::Bool=false,
               skip_weather::Bool=false,
               stream_name::String,
               lag_interval::Number)

Load the historical stream data and possibly live data from a Microprediction.org
stream and combine it with the HRRR weather forecast and NYISO load forecasts to
produce a new TimeArray object.

If a Microprediction.org stream is missing data for a time interval it is
imputed using interpolation.

# Arguments

- `forecast_locations`: The weather forecast locations such as `city`, `solar` or
  `wind` to join to the Microprediction stream history.
- `zscore_features`: Should the returned feature columns be converted to z-scores.
- `load_live_data`: Load the live data from the stream.
- `skip_weather`: Skip joining the weather data to the stream.
- `stream_name`: The Microprediction stream name to load.
- `lag_interval`: The number of intervals to lag the value from the stream, this is
useful to specify the number of intervals in the future to forecast.

"""
function loadStream(;
    forecast_locations="city",
    zscore_features::Bool=true,
    load_live_data::Bool=false,
    skip_weather::Bool=false,
    skip_nyiso::Bool=false,
    stream_name::String,
    lag_interval::Number=1,
    outlier_limit::Float64=0.3,
    additional_streams::Array{String,1}=convert(Array{String,1}, []),
    )




    stream = MicropredictionHistory.loadStream(MICROPREDICTION_HISTORY_DIRECTORY,
            stream_name,
            load_live_data=load_live_data)

    # Since the TimeArray may not contain all of the values since the stream may stop ticking,
    # lets make sure it does, so the values will be impute.

    last_value = missing
    function filter_outliers(x)
        if ismissing(x)
            return x
        end
        # If the stream change is larger than the outlier limit it may make sense
        # to impute a new value for the stream.
        if !ismissing(last_value) && abs(x - last_value) / last_value > outlier_limit
            return missing
        end
        last_value = x
        return x
    end

    stream_values = values(stream.data)
    if match(r"^demand", stream_name) !== nothing
        stream_values = map(filter_outliers, stream_values)
    end
    stream_values = Impute.interpolate(stream_values)

    stream = TimeArray(timestamp(stream.data), convert(Array{Float64,1}, stream_values), ["Demand"])

    # Only deal with stream data after the start date.
    stream = from(stream, MICROPREDICTION_HISTORY_START_DATE)

    if skip_nyiso == false
        # Load the existing ISO forecast.
        iso_forecasts = loadNYISOLoadForecasts()
        stream = merge(stream, lag(iso_forecasts, lag_interval), :inner)
    end


    # If there are additional streams indicated add them.
    for additional_stream_name in additional_streams
        additional_stream = MicropredictionHistory.loadStream(MICROPREDICTION_HISTORY_DIRECTORY,
            additional_stream_name,
            load_live_data=load_live_data)

        additional_stream_values = values(additional_stream.data)
        additional_stream_values = Impute.interpolate(additional_stream_values)

        additional_stream = TimeArray(timestamp(additional_stream.data), convert(Array{Float64,1}, additional_stream_values), ["stream_$(additional_stream_name)"])

        # Only deal with stream data after the start date.
        additional_stream = from(additional_stream, MICROPREDICTION_HISTORY_START_DATE)

        # Merge the lagged additional stream values.
        stream = merge(stream, lag(additional_stream, lag_interval))
    end


    # Now pair that stream with all of the other measured forecasts.

    if skip_weather == false
        # Load all of the weather forecasts from the archive.
        weather_forecasts = loadHRRRForecasts(forecast_locations)

        # Since the weather forecast data is hourly we need to join it again the main data but in
        # a way that the timestamp goes to the last hourly forecast, so build an array that is the
        # same length as the stream updates but contains the timestamp truncated to the hour.

        weather_feature_values::Dict{Symbol,Any} = Dict()
        weather_feature_values[:datetime] = timestamp(stream)

        # Lookup the needed weather features
        # for now only add weather features that contain the word heat_index.
        weather_forecast_feature_columns = filter(x -> true, colnames(weather_forecasts))

        # Lag the weather features by the forecast interval as well.
        merged_stream = merge(stream, lag(weather_forecasts, lag_interval))
    else
        merged_stream = stream
    end

    # Lag the time series by the requested forecast interval and store the column.
    # as demand lag.
    if lag_interval > 0
        m3 = rename(lag(merged_stream[:Demand], lag_interval, padding=true), :Demand_lag)
        merged_stream = merge(merged_stream, m3)

        m3 = rename(merged_stream[:Demand] .- merged_stream[:Demand_lag], :Demand_diff)
        merged_stream = merge(merged_stream, m3)

    end


    # Add the fourier terms with the daily periods and weekly periods based on the stream
    # update interval.  The NYISO streams update every 5 minutes so there are 288 updates
    # a day and 2016 updates a week.
    periodic_range = 1:length(timestamp(merged_stream))

    periodic_values = hcat(
        map(x -> sin(2 * pi * x / 288), periodic_range),
        map(x -> cos(2 * pi * x / 288), periodic_range),
        map(x -> sin(2 * pi * x / 2016), periodic_range),
        map(x -> cos(2 * pi * x / 2016), periodic_range)
    )
    periodic_column_names = [:sin_288, :cos_288, :sin_2016, :cos_2016]

    periodic_values = TimeArray(timestamp(merged_stream), periodic_values, periodic_column_names)

    merged_stream = merge(merged_stream, periodic_values)

    if skip_weather == false
        # Remove rows where the HRRR forecast isn't present, because there wouldn't
        # be enough inputs to the model.
        merged_stream = merged_stream[findwhen(merged_stream[Symbol("hrrr_temperature_0_$(get_location_name(all_locations[forecast_locations][1]))")] .!== missing)]
    end

    if lag_interval > 0
        # Remove rows where the lagged value isn't present, because its a required
        # input to the model.
        merged_stream = merged_stream[findwhen(merged_stream[:Demand_lag] .!== NaN)]
    end

    # Before the features will be passed to the MLP neural network potentially they need
    # to be transformed into z-scores so their scale is consistent.
    #
    # To allow values to be reversed from zscores the mean and the standard deviation
    # should be saved.
    zscore_result::Dict{Symbol,Any} = Dict()
    stats_result::Dict{Symbol,Any} = Dict()
    zscore_result[:datetime] = timestamp(merged_stream)

    for c in colnames(merged_stream)
        s = values(merged_stream[c])
        # Don't change the periodic values to z-scores
        zscore_result[c] = zscore_features && !(c in periodic_column_names) ? StatsBase.zscore(s) : s;
        stats_result[c] = StatsBase.mean_and_std(s)
    end

    return TimeArray(namedtuple(zscore_result); timestamp=:datetime), stats_result
end


# Function to get dictionary of model parameters
function fill_param_dict!(dict, m, prefix)
    if m isa Chain
        for (i, layer) in enumerate(m.layers)
            fill_param_dict!(dict, layer, prefix * "layer_" * string(i) * "/" * string(layer) * "/")
        end
    else
        for fieldname in fieldnames(typeof(m))
            val = getfield(m, fieldname)
            if val isa AbstractArray
                val = vec(val)
            end
            dict[prefix * string(fieldname)] = val
        end
    end
end

struct TrialResult
    name::String
    average_loss::Float64
    best_loss::Float64
end


# These are neural network architectures that will be tried
# when building the final models for parameterized distributions,
# the model that on average produces the lowest loss will be selected.
#
# Add more architectures as you desire.

square(x) = x.^2

architectures = Dict(
    CRPSRegressionSeperate => [
("128-64-64-1-square", function (input_count, activation)
    base_model = Chain(
        Dense(input_count, 128, activation),
        Dense(128, 64, activation),
        Dense(64, 64, activation),
        Dense(64, 64, activation),
        Dense(64, 1),
    )

    distribution_model = Chain(
        Dense(input_count, 128, activation),
        Dense(128, 64, activation),
        Dense(64, 64, activation),
        Dense(64, 64, activation),
        Dense(64, 128, activation),
        Dense(128, length(fixed_quantiles), squre)
     )

    return ((base_model, distribution_model),
        Flux.params(base_model, distribution_model))
end),

    ],
    CRPSRegression => [
          ("128-64-64-64-128", function (input_count, activation)

    model = Chain(
              Dense(input_count, 128, activation),
              Dense(128, 64, activation),
              Dense(64, 64, activation),
              Dense(64, 64, activation),
              Dense(64, 128, activation),
              Dense(128, 1 + length(fixed_quantiles))
          )

    return (model, Flux.params(model))
end),
    ],
    QuantileRegression => [
         ("128-64-64-64-128",
         function (input_count, activation)
    model = Chain(
             Dense(input_count, 128, activation),
             Dense(128, 64, activation),
             Dense(64, 64, activation),
             Dense(64, 64, activation),
             Dense(64, 128, activation),
             Dense(128, 1 + length(fixed_quantiles))
         );
    return (model, Flux.params(model))
end),
    ],
    ClosestPoint => [
        ("100-100",
        function (input_count, activation)
    model = Chain(
            Dense(input_count, 128, activation),
            Dense(128, 64, activation),
            Dense(64, 64, activation),
            Dense(64, 1 + length(fixed_quantiles))
        );
    return (model, Flux.params(model))
end),
    ],
    ParameterizedDistributionDiff => [
        ("128-64-32-2",
        function (input_count, activation)
    model = Chain(
            Dense(input_count, 128, activation),
            Dense(128, 64, activation),
            Dense(64, 32, activation),
            Dense(32, 2)
        );
    return (model, Flux.params(model))
end)
    ],
)



"""

    buildModel(;
        stream_name,
        forecast_locations,
        regressors,
        lag_interval,
        [trial_count,
        learning_rates,
        activations,
        max_epochs,
        batch_size,
        save_filename_prefix])

Build an actual production ready model while iterating through
possible architectures, learning rates, regularization amounts
and activation functions.

The regressors and data used to train the model are the same
across different configurations.

# Arguments

- `stream_name`: The Microprediction stream name which the model
will attempt to forecast
- `forecast_locations`: The identifier of the weather data locations
which will be joined to the stream value.
- `regressors`: An array of regressors to use in the model
- `lag_interval`: The number of intervals ahead that the forecast
will be used for.
- `trial_count`: The number of trials for each model configuration
and architecture.  This will help solve the randomness inherent in
initialization of the network.
- `learning_rates`: An array of learning rates to try
- `activations`: An array of activation functions to try.
- `max_epochs`: The maximum number of epochs to train the model for.
- `batch_size`: The size of the batches used when training the model.
- `lag_interval`: The number of intervals ahead that the forecast
will be used for.
- `save_filename_prefix`: The prefix of the saved BSON files.
- `distribution`: The base distribution type of the model.

"""
function buildModel(;
    stream_name::String,
    forecast_locations::String,
    regressors_by_lag_interval::Dict{Int64,Array{Symbol,1}},
    trial_count::Number=1,
    optimizer_names::Array{String,1}=["ADAM"],
    learning_rates::Array{Float64,1}=[0.001],
    activations=[relu],
    model_approach::ModelApproach,
    max_epochs=1000,
    batch_sizes::Array{Int64,1}=[256],
    additional_streams::Array{String,1}=[],
    save_filename_prefix::String)

    viewize(x) = map(idx -> view(x, idx, :), 1:size(x, 1))

    results_by_lag::Dict{Int64,Dict} = Dict()

    stats_by_lag = Dict()
    stream_start_by_lag = Dict()
    parameterized_distributions = Dict()

    distribution_for_lag(lag_interval) = haskey(parameterized_distributions, lag_interval) ? parameterized_distributions[lag_interval] : missing

    for (lag_interval, regressors) in regressors_by_lag_interval

        stream = loadStream(stream_name=stream_name,
                            zscore_features=true,
                            additional_streams=additional_streams,
                            forecast_locations=forecast_locations,
                            lag_interval=lag_interval)

        if model_approach == ParameterizedDistributionDiff
            # Determine the best distribution to fit.
            parameterized_distributions[lag_interval] = parameterizedDistribution(values(stream[1][:Demand_diff]))
        end

        stream_start_by_lag[lag_interval] = timestamp(stream[1])[1]

        stats_by_lag[lag_interval] = stream[2]

        predicted_field_count = 1;

        if model_approach === QuantileRegression ||
               model_approach === ClosestPoint ||
               model_approach === CRPSRegression ||
               model_approach === CRPSRegressionSeperate
            data_columns = [regressors..., :Demand]
            source_data = convert(Array{Float32}, values(stream[1][data_columns...]))
        else
            data_columns = [regressors..., :Demand_diff]
            source_data = convert(Array{Float32}, values(stream[1][data_columns...]))
        end


        source_data = shuffleobs(source_data, obsdim=1)
        train, test = splitobs(source_data, at=0.3, obsdim=1);

        # The prediction variable is at the end, but may be one or two values.
        train_x = train[:, 1:end - predicted_field_count]
        train_y = train[:, end - (predicted_field_count - 1):end]

        test_x = test[:, 1:end - predicted_field_count]
        test_y = test[:, end - (predicted_field_count - 1):end]

        # Observations are assumed to be the last dimensions, so
        # flip the data around.
        train_x = copy(train_x')
        train_y = copy(train_y')

        test_x = copy(test_x')
        test_y = copy(test_y')


#        println("Training x: ", size(train_x))
#        println("Training y: ", size(train_y))
#        println(typeof(train_y))

        for learning_rate in learning_rates
            for batch_size in batch_sizes
                for (architecture, model_builder) in architectures[model_approach]
                    for optimizer_name in optimizer_names
                        for activation in activations

                            for index in 1:trial_count

                                model_name = "lag=$(lag_interval)-batch=$(batch_size)-o=$(optimizer_name)-arch=$(architecture)-lr=$(learning_rate)-t=$index"

                                training_loader = Flux.Data.DataLoader(train_x, train_y, batchsize=batch_size, shuffle=true)
                                test_loader = Flux.Data.DataLoader(test_x, test_y, batchsize=batch_size, shuffle=true)

                                f = @spawn trainModel(
                                model_name=model_name,
                                regressors=regressors,
                                model_builder=model_builder,
                                training_loader=training_loader,
                                test_loader=test_loader,
                                activation=activation,
                                distribution=distribution_for_lag(lag_interval),
                                epochs=max_epochs,
                                optimizer_name=optimizer_name,
                                learning_rate=learning_rate,
                                model_approach=model_approach,
                                early_stopping_limit=250,
                                batch_size=batch_size)

                                if !haskey(results_by_lag, lag_interval)
                                    results_by_lag[lag_interval] = Dict()
                                end

                                if !haskey(results_by_lag[lag_interval], model_name)
                                    results_by_lag[lag_interval][model_name] = []
                                end
                                push!(results_by_lag[lag_interval][model_name], f)
                            end
                        end
                    end
                end
            end
        end
    end

    println("Getting results")

    report = []

    best_models_by_lag = Dict()
    # Determine the best model for each lag interval.
    for (lag_interval, model_suite) in results_by_lag


        sorted_models_by_loss = []


        # There can be multiple trials of the same model configuration so find
        # the best one.

        for (model_name, training_results) in model_suite
            sorted_results::Array{ModelTrainResult,1} = sort(map(fetch, training_results), by=x -> x.best_test_loss)
            average_loss = mean(map(x -> x.best_test_loss, sorted_results))
            average_epoch = mean(map(x -> x.epoch, sorted_results))
            push!(sorted_models_by_loss, Dict(
                :average_loss => average_loss,
                :name => model_name,
                :training_result => sorted_results[1],
                :average_epoch => average_epoch))
        end

        sorted_models_by_loss = sort(sorted_models_by_loss, by=x -> x[:average_loss])


        training_losses = []
        push!(report, "Lag Interval: $(lag_interval)")
        for result in sorted_models_by_loss
            push!(report, "$(result[:average_loss])\t$(result[:training_result].name)\t$(result[:average_epoch])")
            push!(training_losses, Dict(:name => result[:training_result].name,
                                        :loss_history => result[:training_result].losses_by_epoch))
        end

        full_save_filename = "$(save_filename_prefix)-lag-$(lag_interval).binary"
        println("Saving to: $(full_save_filename)")

        top_model = sorted_models_by_loss[1][:training_result]

        save_data = Dict(:model => top_model.model,
                         :model_name => top_model.name,
                         :training_losses => training_losses,
                         :all_results => sorted_models_by_loss,
                         :lag_interval => lag_interval,
                         :regressors => top_model.regressors,
                         :model_approach => model_approach,
                         :distribution => distribution_for_lag(lag_interval),
                         :stream => stream_name,
                         :report => report,
                         :forecast_locations => forecast_locations,
                         :stream_start => stream_start_by_lag[lag_interval],
                         :stats => stats_by_lag[lag_interval])
        serialize(full_save_filename, save_data)
        println("Finished")

        best_models_by_lag[lag_interval] = save_data
    end

    map(println, report)

    return best_models_by_lag
end

struct ModelTrainResult
    name::String
    regressors::Array{Symbol,1}
    model
    best_test_loss::Union{Missing,Float32}
    epoch::Number
    losses_by_epoch::Array{Dict{Symbol,Float64},1}
    batch_size::Int64
end


"""
    trainModel(
        model_name::String,
        regressors::Array{Symbol,1},
        model_builder,
        training_loader::Flux.Data.DataLoader,
        test_loader::Flux.Data.DataLoader,
        epochs::UInt=10,
        early_stopping_limit::UInt=10,
        learning_rate=0.001,
        distribution,
        activation=elu
    )

Train a model for a specified number of epochs and stopping early
if the model's loss on the test set does not improve over a fixed
number of epochs.

# Arguments

- `activation`: The activation function used by the model.
- `distribution`: The distribution that being fit by the model's output.
- `early_stopping_limit`: If loss does not improve on the test set for the number epochs specified, training stops.
- `epochs`: The maximum number of epochs to train the model
- `learning_rate`: The learning rate to use with the optimizer.
- `model_builder`: A function that builds the model's architecture
- `model_name`: The name used to identify the model.
- `regressors`: A list of regressor names used by the model.
- `test_loader`: A DataLoader that supplies the test data
- `training_loader`: A DataLoader that supplies the training data

"""
function trainModel(;
    activation=elu,
    distribution,
    early_stopping_limit::Number=25,
    epochs::Number=10,
    batch_size::Int64,
    learning_rate::Float64,
    optimizer_name::String,
    model_approach::ModelApproach,
    model_builder,
    model_name,
    regressors::Array{Symbol,1},
    test_loader::Flux.Data.DataLoader,
    training_loader::Flux.Data.DataLoader,
    )::ModelTrainResult

    # println("Train loader X size:", size(training_loader.data[1]))
    # println("Train loader Y size:", size(training_loader.data[1]))

#    logger = TBLogger("content/$(model_name)", tb_overwrite)

    # Determine the number of inputs to the model by looking at the
    # first training example, since there can be a variable number of
    # regressors used.
    input_count::Int32 = convert(Int32, size(training_loader.data[1], 1))

    # println("TL:", size(training_loader.data[1]))
    # println("Input count:", input_count)
    # Build the actual model.
    model, model_params = model_builder(input_count, activation)

    if optimizer_name === "ADAM"
        optimizer = ADAM(learning_rate)
    elseif optimizer_name === "Descent"
        optimizer = Descent(learning_rate)
    elseif optimizer_name === "Momentum"
        optimizer = Momentum(learning_rate)
    elseif optimizer_name === "AMSGrad"
        optimizer = AMSGrad(learning_rate)
    elseif optimizer_name === "RADAM"
        optimizer = RADAM(learning_rate)
    elseif optimizer_name === "ADAMW"
        optimizer = ADAMW(learning_rate)
    elseif optimizer_name === "NADAM"
        optimizer = NADAM(learning_rate)
    elseif optimizer_name === "ADADelta"
        optimizer = ADADelta(learning_rate)
    elseif optimizer_name === "ADAGrad"
        optimizer = ADAGrad(learning_rate)
    elseif optimizer_name === "AdaMax"
        optimizer = AdaMax(learning_rate)
    elseif optimizer_name === "RMSProp"
        optimizer = RMSProp(learning_rate)
    else
        throw(ErrorException("Unknown optimizer: $(optimizer_name)"))
    end
#    optimizer = ADAM(learning_rate)
    decay = Flux.Optimise.ExpDecay(0.01, 0.90, 25, 0.001)

 #   optimizer = Flux.Optimise.Optimiser(decay, ADAM(0.01))

    tracked_losses::Array{Dict{Symbol,Float64},1} = []

    # Track that loss is decreasing otherwise stop early.
    early_stopping_counter = 0
    best_test_loss::Union{Missing,Float64} = missing

    best_model = nothing
    best_epoch = 0
    last_epoch = 0

    local loss
    local big_loss

    first = true
    if model_approach === ParameterizedDistributionDiff
        loss = (y1, y2) -> generic_loss_parameterized_distribution(model, distribution, y1, y2)
        big_loss = (y1, y2) -> loss_total_parameterized_distribution(model, distribution, y1, y2)
    elseif model_approach === QuantileRegression
        loss = (y1, y2) -> generic_loss_quantile(model, y1, y2)
        big_loss = (y1, y2) -> loss_total_quantile(model, y1, y2)
    elseif model_approach === CRPSRegression
        loss = (y1, y2) -> generic_loss_crps(model, y1, y2)
        big_loss = (y1, y2) -> loss_total_crps(model, y1, y2)
    elseif model_approach === CRPSRegressionSeperate
        loss = (y1, y2) -> generic_loss_crps_seperate(model, y1, y2)
        big_loss = (y1, y2) -> loss_total_crps_seperate(model, y1, y2)
    elseif model_approach === ClosestPoint
        loss = (y1, y2) -> generic_loss_closest_point(model, y1, y2)
        big_loss = (y1, y2) -> loss_total_closest_point(model, y1, y2)
    else
        throw(ErrorException("Unknown model approach loss function"))
    end

    println("Starting training loop")
    try
        for i in 1:epochs
            last_epoch = i

            # The data loader won't return the inputs batched together for
            # efficient compution, take care of this by concatenating the
            # batches into a matrix rather than an array of arrays.

#        Flux.trainmode!(model)
            train_time = @elapsed Flux.train!(loss, model_params, training_loader, optimizer)
        #        Flux.testmode!(model)

            test_loss = big_loss(test_loader.data[1], test_loader.data[2])
            training_loss = big_loss(training_loader.data[1], training_loader.data[2])


            push!(tracked_losses, Dict(
            :training_total => training_loss[:total],
            :test_total => test_loss[:total],
            :training_regularization => training_loss[:reg_loss],
            :test_regularization => test_loss[:reg_loss],
            :learning_rate => decay.eta,
        ))


            if last_epoch % 10 == 0
                foo = @sprintf "%30s\t%4d\ttest=%.4f\ttrain=%.4f\tlead=%.2f\treg=%.2f\tt=%.3f\n" model_name last_epoch test_loss[:total] training_loss[:total] test_loss[:total] - training_loss[:total] test_loss[:reg_loss] train_time
                print(foo)
            end

            # If logging to tensorboard use this.

            # Implement some early stopping, persist the model with the best
            # loss on the test set so far.
            if last_epoch > 1
                if best_test_loss !== missing && best_test_loss < test_loss[:total]
                    early_stopping_counter = early_stopping_counter + 1
                elseif (best_test_loss === missing || best_test_loss > test_loss[:total])
                    best_test_loss = test_loss[:total]
                    best_epoch = last_epoch
                    best_model = deepcopy(model)
                    early_stopping_counter = 0
                end
            end

            if early_stopping_counter == early_stopping_limit
                println("Epoch $(i) Stopping since loss not improving $(model_name) $(early_stopping_limit) best $(best_test_loss) current: $(test_loss)")
                break
            end
        end
        if last_epoch == epochs
            println("Reached epoch training limit $(last_epoch) $(model_name)")
        end
    catch e
        println("Error in training: $(e)")
    end
    return ModelTrainResult(
        model_name,
        regressors,
        best_model,
        best_test_loss,
        best_epoch,
        tracked_losses,
        batch_size)
end

"""
    parameterizedDistribution(stream_name, lag_interval)

Compare various distributions for goodness of fit for an array
values.  Returns the distribution that fits the best.

"""
function parameterizedDistribution(values)
    dist = [DistributionsAD.Normal, DistributionsAD.Cauchy, DistributionsAD.Laplace]
    dist_names = ["Normal", "Cauchy", "Laplace"]
    bad = filter(x -> isnan(x) || isinf(x), values)
    ll = []
    for d in dist
        try
            fitted = fit(d, values)
            lv = loglikelihood(fitted, values)
            push!(ll, lv)
        catch
            push!(ll, NaN)
        end
    end

    dist[findmax(ll)[2]]
end


# To accelerate calculating the CRPS across a fixed array of quantiles
# perform some of the computation at compile time and store the results
# as constants.
const fixed_quantiles_minus_1 = convert(Array{Float32}, fixed_quantiles .- 1.0f0)
const fixed_quantiles_minus_1_sq = convert(Array{Float32}, fixed_quantiles_minus_1.^2)
const fixed_quantiles_sq = convert(Array{Float32}, fixed_quantiles.^2)

"""
    crps(quantile_values, actual_values)

Calculate the Continously Ranked Probablity Score for an array
of quantile values and an array of actual value.

# Arguments

- `quantile_values`: A 2d array of quantile values where the quantiles
are stored in the first dimension.
- `actual_values`: A array of observation values.

# Returns

Return the sum total of the CRPS for each example and each set of quantiles.

"""
@inline function crps(quantile_values::AbstractArray{Float32,2}, actual::Array{Float32,1})::Float32
    integral::Float32 = 0.0f0

    total_quantiles::Int64 = size(quantile_values, 1)

    @inbounds for batch_index in 1:size(quantile_values, 2)
        obs_cdf::Bool = false

        @inbounds av::Float32 = actual[batch_index]
        @inbounds qv = view(quantile_values, :, batch_index)

        previous_forecast::Float32 = 0.0f0
        @inbounds for n in 1:total_quantiles
            @inbounds qvv::Float32 = qv[n]

            if obs_cdf === false && av < qvv
                integral += ((av - previous_forecast) * fixed_quantiles_sq[n]) + ((qvv - av) * (fixed_quantiles_minus_1_sq[n]))
                obs_cdf = true
            else
                integral += ((qvv - previous_forecast)) * (obs_cdf ? fixed_quantiles_minus_1_sq[n] : fixed_quantiles_sq[n])
            end
            previous_forecast = qvv
        end
        if obs_cdf === false
            @inbounds integral += av - previous_forecast
        end
    end
    return integral
end
