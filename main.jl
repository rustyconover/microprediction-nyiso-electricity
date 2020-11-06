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
using Logging
using MicropredictionHistory
using NamedTupleTools
using OhMyREPL
using Plots
using RCall
using Serialization
using StatsBase
#using TensorBoardLogger
using TimeSeries
using CSV
using TimeZones
using MLDataUtils
using BSON
using JSON

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
    Dict("name" => "Relative humidity", "level" => 0, "typeOfLevel" => "unknown"),
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

# Where should the forecast files be stored.
FORECAST_DIRECTORY = "/Users/rusty/Data/weather-forecasts/"

# The directory where the Microprediction.org stream history is stored.
MICROPREDICTION_HISTORY_DIRECTORY = "/Users/rusty/Development/pluto/data"

# The date at which Microprediction.org history became reliable.
MICROPREDICTION_HISTORY_START_DATE = DateTime(2020, 9, 15, 0, 0, 0)

# The directory where NYISO forecasts should be stored.
NYISO_FORECAST_DIRECTORY = "/Users/rusty/Data/nyiso-load/"


LIVE_FORECAST_DIRECTORY = "/Users/rusty/Data/live-weather-forecasts/"

include("bounds.jl")
include("grib.jl")

include("nyiso-forecast.jl")
include("hrrr.jl")

include("regularized-dense-layer.jl")

include("prediction.jl")
include("feature-selection.jl")


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
    stream_name::String,
    lag_interval::Number=1,
    )

    R"""library(imputeTS)"""

    stream = MicropredictionHistory.loadStream(MICROPREDICTION_HISTORY_DIRECTORY,
            stream_name,
            load_live_data=load_live_data)

    # Since the TimeArray may not contain all of the values since the stream may stop ticking,
    # lets make sure it does, so the values will be imputed by R.

    stream_values = values(stream.data)
    R"""
imputed <- na_interpolation($stream_values)
"""

    @rget imputed

    stream = TimeArray(timestamp(stream.data), convert(Array{Float64,1}, imputed), ["Demand"])

    # Only deal with stream data after the start date.
    stream = from(stream, MICROPREDICTION_HISTORY_START_DATE)

    # Load the existing ISO forecast.
    iso_forecasts = loadNYISOLoadForecasts()

    stream = merge(stream, lag(iso_forecasts, lag_interval), :inner)

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
    end

    # Lag the time series by the requested forecast interval and store the column.
    # as demand lag.
    m3 = rename(lag(merged_stream[:Demand], lag_interval, padding=true), :Demand_lag)
    merged_stream = merge(merged_stream, m3)

    m3 = rename(merged_stream[:Demand] .- merged_stream[:Demand_lag], :Demand_diff)
    merged_stream = merge(merged_stream, m3)

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

    # Remove rows where the HRRR forecast isn't present, because there wouldn't
    # be enough inputs to the model.
    merged_stream = merged_stream[findwhen(merged_stream[Symbol("hrrr_temperature_0_$(get_location_name(all_locations[forecast_locations][1]))")] .!== missing)]

    # Remove rows where the lagged value isn't present, because its a required
    # input to the model.
    merged_stream = merged_stream[findwhen(merged_stream[:Demand_lag] .!== NaN)]

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
# when building the final models, the model that on average
# produces the lowest loss will be selected.
#
# Add more architectures as you desire.
model_architectures = [
    ("64l1-64-32", (input_count, activation, l1_regularization, l2_regularization) ->
    Chain(
        RegularizedDense(Dense(input_count, 64, activation), l1_regularization, l2_regularization),
        RegularizedDense(Dense(64, 64, activation), 0, 0),
        RegularizedDense(Dense(64, 32, activation), 0, 0),
        Dense(32, 2)
    )),
    ("64l1-64-d0.1-32", (input_count, activation, l1_regularization, l2_regularization) ->
    Chain(
        RegularizedDense(Dense(input_count, 64, activation), l1_regularization, l2_regularization),
        RegularizedDense(Dense(64, 64, activation), 0, 0),
        Dropout(0.1),
        RegularizedDense(Dense(64, 32, activation), 0, 0),
        Dense(32, 2)
    )),
    ("128l1-128-128-128", (input_count, activation, l1_regularization, l2_regularization) ->
    Chain(
        RegularizedDense(Dense(input_count, 128, activation), l1_regularization, l2_regularization),
        RegularizedDense(Dense(128, 128, activation), 0, 0),
        RegularizedDense(Dense(128, 128, activation), 0, 0),
        RegularizedDense(Dense(128, 128, activation), 0, 0),
        Dense(128, 2)
    )),
    ("128l1-64-32", (input_count, activation, l1_regularization, l2_regularization) ->
    Chain(
        RegularizedDense(Dense(input_count, 128, activation), l1_regularization, l2_regularization),
        RegularizedDense(Dense(128, 64, activation), 0, 0),
        RegularizedDense(Dense(64, 32, activation), 0, 0),
        Dense(32, 2)
    )),
    ("256l1-64-32", (input_count, activation, l1_regularization, l2_regularization) ->
    Chain(
        RegularizedDense(Dense(input_count, 256, activation), l1_regularization, l2_regularization),
        RegularizedDense(Dense(256, 64, activation), 0, 0),
        RegularizedDense(Dense(64, 32, activation), 0, 0),
        Dense(32, 2)
    )),
    ("128l1-64-d0.1-64-32", (input_count, activation, l1_regularization, l2_regularization) ->
    Chain(
        RegularizedDense(Dense(input_count, 128, activation), l1_regularization, l2_regularization),
        RegularizedDense(Dense(128, 64, activation), 0, 0),
        Dropout(0.1),
        RegularizedDense(Dense(64, 64, activation), 0, 0),
        RegularizedDense(Dense(64, 32, activation), 0, 0),
        Dense(32, 2)
    )),
    ("128l1-64-d0.05-64-d0.05-32", (input_count, activation, l1_regularization, l2_regularization) ->
    Chain(
        RegularizedDense(Dense(input_count, 128, activation), l1_regularization, l2_regularization),
        RegularizedDense(Dense(128, 64, activation), 0, 0),
        Dropout(0.05),
        RegularizedDense(Dense(64, 64, activation), 0, 0),
        Dropout(0.05),
        RegularizedDense(Dense(64, 32, activation), 0, 0),
        Dense(32, 2)
    ))
];

"""
    build_wind_power(save_filename_prefix, max_epochs, trial_count, lag_intervals)

Build a production solar power model with the passed parameters.

# Arguments

- `save_filename_prefix`: The prefix of the saved BSON files.
- `max_epochs`: The maximum number of epochs the model will be trained for,
  if the model is no longer improving it will be stopped early.
- `trial_count`: The number of trials for each model configuration
  and architecture.  This will help solve the randomness inherent in
  initialization of the network.
- `lag_intervals`: An array of lag intervals for which the network should be trained.

"""
function build_wind_power(;
    save_filename_prefix,
    max_epochs=1000,
    trial_count=1,
    lag_intervals=[1, 3, 12])
    stream_name="electricity-fueltype-nyiso-wind.json"
    forecast_locations="wind"

    # Load the stream so that the symbol names of the regressors can be filtered.
    stream = loadStream(stream_name=stream_name,
        zscore_features=true,
        forecast_locations=forecast_locations,
        lag_interval=lag_intervals[1])

    return buildModel(
        stream_name=stream_name,
        forecast_locations=forecast_locations,
        regressors=[
            ["last_demand", Set([:Demand_lag])],
            ["average_wind_speed", Set(filter(x -> contains(String(x), "average_wind_speed"), colnames(stream[1])))],
            ["relative_humidity", Set(filter(x -> contains(String(x), "relative_humidity"), colnames(stream[1])))],
            ["minimum_wind_speed", Set(filter(x -> contains(String(x), "minimum_wind_speed"), colnames(stream[1])))],
        ],
        max_epochs=max_epochs,
        trial_count=trial_count,
        lag_intervals=lag_intervals,
        save_filename_prefix=save_filename_prefix)
end


function build_demand_stream(;
    stream_name::String,
    save_filename_prefix::String,
    max_epochs::Number=1000,
    trial_count::Number=1)

    forecast_locations="city"


    # Load the stream so that the symbol names of the regressors can be filtered.
    stream = loadStream(stream_name=stream_name,
        zscore_features=true,
        forecast_locations=forecast_locations,
        lag_interval=1)

    summarized_feature_selection = collect(summarizeFeatureSelection(3, 5, [stream_name]))

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

    filtered_regressors::Dict{Number,Array{Symbol,1}} = Dict()
    for (lag_interval, all_feature_names) in feature_selection_preferred
        filtered_regressors[lag_interval] = filter(filter_regressors, all_feature_names)
    end

    println(filtered_regressors)

    return buildModel(
        stream_name=stream_name,
        forecast_locations=forecast_locations,
        regressors_by_lag_interval=filtered_regressors,
        max_epochs=max_epochs,
        trial_count=trial_count,
        save_filename_prefix=save_filename_prefix)

end




"""
    build_solar_power(save_filename_prefix, max_epochs, trial_count, lag_intervals)

Build a production solar power model with the passed parameters.

# Arguments

- `save_filename_prefix`: The prefix of the saved BSON files.
- `max_epochs`: The maximum number of epochs the model will be trained for,
  if the model is no longer improving it will be stopped early.
- `trial_count`: The number of trials for each model configuration
  and architecture.  This will help solve the randomness inherent in
  initialization of the network.
- `lag_intervals`: An array of lag intervals for which the network should be trained.

"""
function build_solar_power(;
    save_filename_prefix,
    max_epochs=1000,
    trial_count=1,
    lag_intervals=[1, 3, 12])

    stream_name="electricity-fueltype-nyiso-other_renewables.json"
    forecast_locations="solar"

    # Load the stream so that the symbol names of the regressors can be filtered.
    stream = loadStream(stream_name=stream_name,
        zscore_features=true,
        forecast_locations=forecast_locations,
        lag_interval=lag_intervals[1])


    summarized_feature_selection = collect(summarizeFeatureSelection(3, 5, [stream_name]))

    feature_selection_preferred = Dict(
        map(x -> Pair(x[1], collect(union(regressor_names_to_columns(x[2], stream[1])...))),
        summarized_feature_selection))


    return buildModel(
        stream_name=stream_name,
        forecast_locations=forecast_locations,
        regressors_by_lag_interval=feature_selection_preferred,
        max_epochs=max_epochs,
        trial_count=trial_count,
        save_filename_prefix=save_filename_prefix)
end


"""

    buildModel(;
        stream_name, forecast_locations, regressors,
        lag_interval, [trial_count, learning_rates,
        l1_regularizations, activations, max_epochs,
        batch_size, save_filename_prefix])

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
- `l1_regularizations`: An array of l1 regularization amounts to try
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
    learning_rates::Array{Float64,1}=[0.001],
    l1_regularizations::Array{Float64,1}=[0, 0.01, 0.05],
    activations=[gelu],
    max_epochs=1000,
    batch_size=32,
    save_filename_prefix::String)

    viewize(x) = map(idx -> view(x, idx,:), 1:size(x,1))

    results_by_lag::Dict{Int64,Dict{String,Array{Future,1}}} = Dict()


    stats_by_lag = Dict()

    stream_start_by_lag = Dict()

    parameterized_distributions = Dict()

    for (lag_interval, regressors) in regressors_by_lag_interval

        stream = loadStream(stream_name=stream_name,
                            zscore_features=true,
                            forecast_locations=forecast_locations,
                            lag_interval=lag_interval)


        parameterized_distributions[lag_interval] = parameterizedDistribution(values(stream[1][:Demand_diff]))

        stream_start_by_lag[lag_interval] = timestamp(stream[1])[1]

        stats_by_lag[lag_interval] = stream[2]

        data_columns = [regressors..., :Demand_diff]

        source_data = convert(Array{Float32}, values(stream[1][data_columns...]))
        source_data = shuffleobs(source_data, obsdim=1)
        train, test = splitobs(source_data, at = 0.3, obsdim=1);

        # The prediction variable :Demand is at the end.
        train_x = train[:, 1:end-1]
        train_y = train[:, end]

        test_x = test[:, 1:end-1]
        test_y = test[:, end]


        for learning_rate in learning_rates
            for (architecture, model_builder) in model_architectures
                for l1_regularization in l1_regularizations
                    for activation in activations
                        model_name = "activation=$(activation)-l1=$(l1_regularization)-arch=$(architecture)-lr=$(learning_rate)"

                        println(model_name)

                        for index in 1:trial_count

                            training_loader = Flux.Data.DataLoader(viewize(train_x), train_y, batchsize=batch_size, shuffle=true)
                            test_loader = Flux.Data.DataLoader(viewize(test_x), test_y, batchsize=batch_size, shuffle=true)

                            f = @spawn trainModel(
                                    model_name=model_name,
                                    regressors=regressors,
                                    model_builder=model_builder,
                                    training_loader=training_loader,
                                    test_loader=test_loader,
                                    activation=activation,
                                    distribution=parameterized_distributions[lag_interval],
                                    epochs=max_epochs,
                                    learning_rate=learning_rate,
                                    early_stopping_limit=20,
                                    l1_regularization=l1_regularization,
                                    l2_regularization=0.0)

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

    println("Getting results")
    for (lag, model_suite) in results_by_lag
        for (model_name, model) in model_suite
            for r in model
                @async fetch(r)
            end
        end
    end

    report = []

    best_models_by_lag = Dict()
    # Determine the best model for each lag interval.
    for (lag_interval, model_suite) in results_by_lag
        actual_results = []
        for (name, results) in model_suite
            sorted_results::Array{ModelTrainResult,1} = sort(map(fetch, results), by=x -> x.best_test_loss)
            average_loss = mean(map(x -> x.best_test_loss, sorted_results))
            average_epoch = mean(map(x -> x.epoch, sorted_results))

            push!(actual_results, (average_loss, sorted_results[1], average_epoch))
        end

        actual_results = sort(actual_results, by=x -> x[1])
        push!(report, "Lag Interval: $(lag_interval)")
        for (average_loss, best_model, average_epoch) in actual_results
            push!(report, "$(average_loss)\t$(best_model.name)\t$(average_epoch)")
        end

        model = actual_results[1][2].model
        full_save_filename = "$(save_filename_prefix)-lag-$(lag_interval).binary"
        println("Saving to: $(full_save_filename)")
        save_data = Dict(:model => actual_results[1][2].model,
                         :lag_interval => lag_interval,
                         :regressors => actual_results[1][2].regressors,
                         :distribution => parameterized_distributions[lag_interval],
                         :stream => stream_name,
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
    model::Union{Chain,Nothing}
    best_test_loss::Union{Missing,Float32}
    epoch::Number
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
        l1_regularization,
        l2_regularization,
        activation=gelu
    )

Train a model for a specified number of epochs and stopping early
if the model's loss on the test set does not improve over a fixed
number of epochs.

# Arguments

- `model_name`: The name used to identify the model.
- `regressors`: A list of regressor names used by the model.
- `model_builder`: A function that builds the model's architecture
- `training_loader`: A DataLoader that supplies the training data
- `test_loader`: A DataLoader that supplies the test data
- `epochs`: The maximum number of epochs to train the model
- `early_stopping_limit`: If loss does not improve on the test set
for the number epochs specified, training stops.
- `learning_rate`: The learning rate to use with the optimizer.
- `distribution`: The distribution that being fit by the model's
output.
- `l1_regularization`: The amount of l1 regularlization to use.
- `l2_regularization`: The amount of l2 regularlization to use.
- `activation`: The activation function used by the model.

"""
function trainModel(;
    model_name,
    regressors::Array{Symbol,1},
    model_builder,
    training_loader::Flux.Data.DataLoader,
    test_loader::Flux.Data.DataLoader,
    epochs::Number=10,
    early_stopping_limit::Number=10,
    learning_rate=0.001,
    distribution,
    l1_regularization=0.0,
    l2_regularization=0.0,
    activation=gelu)::ModelTrainResult

#    logger = TBLogger("content/$(model_name)", tb_overwrite)

    # Determine the number of inputs to the model by looking at the
    # first training example, since there can be a variable number of
    # regressors used.
    input_count = size(training_loader.data[1][1], 1)

    # Build the actual model.
    model = model_builder(input_count, activation, l1_regularization, l2_regularization)

    LOSS_SMIDGE = Float32(0.0001)

    function loss(ŷ, y)
        model_result = model(ŷ)
        mu = model_result[1, :]

        std = softplus.(model_result[2, :]) .+ LOSS_SMIDGE

        likelihood_loss = -sum(zip(mu, std, y)) do (mu, std, y_target)
            DistributionsAD.logpdf(distribution(mu, std), y_target) + LOSS_SMIDGE
        end

        return likelihood_loss + penalty(model)
    end

    # Calculate the loss but more efficiently than example by example, stack
    # up the data into a large batch.
    function loss_total(ŷ, y)
        x_batch = reduce(hcat, ŷ)
        y_batch = reduce(hcat, y)

        model_result = model(x_batch)
        mu = model_result[1, :]
        std = softplus.(model_result[2, :]) .+ LOSS_SMIDGE

        reg_loss = penalty(model)
        likelihood_loss = -sum(map(x -> DistributionsAD.logpdf(distribution(x[1], x[2]), x[3]) + LOSS_SMIDGE, zip(mu, std, y_batch)))
        return (likelihood_loss + reg_loss, reg_loss)
    end

    # Callback to log information after every epoch to tensorboard.
    function TBCallback(i, training_loss, test_loss, training_loss_reg, test_loss_reg)
        param_dict = Dict{String,Any}()
        fill_param_dict!(param_dict, model, "")

        println("$(model_name) Epoch $(i) test loss: $(test_loss) train $(training_loss)")
        with_logger(logger) do
            @info "model" params = param_dict log_step_increment = 0
            @info "test_reg" loss = test_loss_reg log_step_increment = 0
            @info "test" loss = test_loss
        end
    end

    optimizer = ADAM(learning_rate)
    p = Flux.params(model)

    # Track that loss is decreasing otherwise stop early.
    early_stopping_counter = 0
    best_test_loss::Union{Missing,Float64} = missing

    println("Starting training")

    best_model = nothing
    best_epoch = 0
    last_epoch = 0
    try
        for i in 1:epochs
            last_epoch = i

            # The data loader won't return the inputs batched together for
            # efficient compution, take care of this by concatenating the
            # batches into a matrix rather than an array of arrays.
            x_real = []
            for i in training_loader
                push!(x_real, (reduce(hcat, i[1]), i[2]))
            end

            Flux.train!(loss, p, x_real, optimizer)

            Flux.testmode!(model)
            test_loss, test_loss_reg = loss_total(test_loader.data[1], test_loader.data[2])
            Flux.trainmode!(model)

            println("$(test_loss)\t$(model_name)")

            # If logging to tensorboard use this.
            # TBCallback(i, training_loss, test_loss, training_loss_reg, test_loss_reg)

            # Implement some early stopping, persist the model with the best
            # loss on the test set so far.
            if best_test_loss !== missing && best_test_loss < test_loss
                early_stopping_counter = early_stopping_counter + 1
            else
                if best_test_loss === missing || best_test_loss > test_loss
                    best_test_loss = test_loss
                    best_epoch = last_epoch
                    best_model = deepcopy(model)
                    early_stopping_counter = 0
                end
            end

            if early_stopping_counter == early_stopping_limit
                println("Epoch $(i) Stoping since loss not improving $(model_name) $(early_stopping_limit) best $(best_test_loss) current: $(test_loss)")
                break
            end
        end
        if last_epoch == epochs
            println("Reached epoch training limit")
        end
    catch e
        println("Error in training: $(e)")
    end
    return ModelTrainResult(model_name, regressors, best_model, best_test_loss, best_epoch)
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
            lv =loglikelihood(fitted, values)
            push!(ll, lv)
        catch
            push!(ll, NaN)
        end
    end

    dist[findmax(ll)[2]]
end

