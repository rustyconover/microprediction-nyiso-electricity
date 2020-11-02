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
    :temperature => cols_starting("temperature"),
    :dewpoint_temperature => cols_starting("2_metre_dewpoint"),
    :average_wind_speed => cols_containing("average_wind_speed"),
    :maximum_wind_speed => cols_containing("maximum_wind_speed"),
    :minimum_wind_speed => cols_containing("minimum_wind_speed"),
    :wind_components => cols_containing("wind_component"),

    :total_cloud_cover => cols_containing("total_cloud_cover"),
    :low_cloud_cover => cols_containing("low_cloud_cover"),
    :high_cloud_cover => cols_containing("high_cloud_cover"),
    :medium_cloud_cover => cols_containing("medium_cloud_cover"),
    :visible_diffuse_downward_solar_flux => cols_containing("visible_diffuse_downward"),
    :visible_beam_downward_solar_flux => cols_containing("visible_beam_downward"),
)



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

struct FeatureComparisionResult
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
    combo_lengths,
    trial_count::Number=16,
    distribution,
    early_stopping_limit::Number=20,
    epochs::Number=1000,
    learning_rate::Float64=0.01)::Array{FeatureComparisionResult,1}
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
        all_regressors = sort(collect(union(idea_column_names...)))

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

        training_loader = Flux.Data.DataLoader(viewize(train_x), train_y, batchsize=32, shuffle=true)
        test_loader = Flux.Data.DataLoader(viewize(test_x), test_y, batchsize=32, shuffle=true)

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

    #     push!(results,
    #         FeatureComparisionResult(
    #            trials[1].regressors,
    #             average_loss,
    #             minimum_loss))
    end

    results = sort(results, by=x -> x.average_loss)

    println("Summary:")
    println("-------------------------------")
    for record in results
        @printf "%10.3f\t%10.3f\t%s\n" record.average_loss record.minimum_loss join(map(String, record.regressors), "-")
    end
    return results
end
