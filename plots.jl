function plot_regressor_distributions()
    # Build a violin Plots
    forecasts = loadStream()[1]

    p = plot(size=(1200, 1200), title="Distributions of Regressor Variables", legend=:outertopright)
    for name in filter(x -> x !== :lagged_A && x !== :A, sort(colnames(forecasts)))
        violin!(p, values(forecasts[name]), show_mean=true, show_median=true, label=String(name))
    end
    return p
end


function plot_streams_load(name)
    electricity_load_streams = collect(filter(x -> startswith(x, "electricity-load-nyiso-overall"), keys(Microprediction.get_sponsors(Microprediction.Config()))))

    plots = map(function (stream_name)
        f = loadStream(stream_name=stream_name)

        f = (from(f[1], DateTime(2020, 10, 9, 0, 0, 0)), f[2])
        names = filter(x -> startswith(String(x), name), colnames(f[1]))
        p = plot()

        for name in names
            plot!(p, f[1][Symbol(name)], label=replace(String(name), "temperature_0_" => ""), legend=:outertopright)
        end

        plot!(p, f[1][:Demand], title=stream_name, label="Overall Demand", legend=:outertopright, ylim=(-3, 4), color="black", width=2)

        return p
    end, electricity_load_streams[1:1])

    big_p = plot(plots..., size=(1800, 500), layout=(size(plots, 1), 1), title="Temperature vs Overall Demand")
    return big_p
end

function plot_streams_load_coor(name)
    electricity_load_streams = collect(filter(x -> startswith(x, "electricity-load-nyiso-overall"), keys(Microprediction.get_sponsors(Microprediction.Config()))))

    f = loadStream(stream_name=electricity_load_streams[1])
    names = filter(x -> startswith(String(x), name), colnames(f[1]))[1:1]

    corrplot(values(f[1][:Demand, names...]), label=["Demand", names...], size=(2000, 2000), grid=false)
    savefig("/Users/rusty/Desktop/big-coor.png")
    return
end

function plot_streams_load_coor_scatter(name)
    electricity_load_streams = collect(filter(x -> startswith(x, "electricity-load-nyiso-overall"), keys(Microprediction.get_sponsors(Microprediction.Config()))))

    f = loadStream(stream_name=electricity_load_streams[1])
    names = filter(x -> startswith(String(x), name), colnames(f[1]))

    p = plot(size=(1700, 500), legend=:outertopright)
    for n in names
        scatter!(p, values(f[1][:Demand]), values(f[1][n]), label=replace(String(n), "temperature_0_" => ""), xlabel="Demand", ylabel=name, markersize=1, markerstrokewidth=0, alpha=0.5)
    end
    savefig("/Users/rusty/Desktop/big-coor.png")
    return
end



function plot_streams_fuel_type()
    electricity_load_streams = collect(filter(x -> startswith(x, "electricity-fueltype"), keys(Microprediction.get_sponsors(Microprediction.Config()))))

    plots = map(function (stream_name)
        f = loadStream(stream_name=stream_name)
        p = plot(f[1][:Demand], title=stream_name, label="Electricity Demand", legend=:outertopright, ylim=(-3, 3))

        plot!(p, f[1][:heat_index_0_1], label="Heat Index Location 1")
        plot!(p, f[1][:relative_humidity_0_1], label="Relative Humidity Location 1")
        plot!(p, f[1][:average_wind_speed_10_1], label="Average Wind Speed")

    end, electricity_load_streams)

    big_p = plot(plots..., size=(1800, 3000), layout=(size(plots, 1), 1))
    return big_p
end

function plot_streams_fuel_type_wind_combined(name)
    electricity_load_streams = collect(filter(x -> startswith(x, "electricity-fueltype-nyiso-wind"), keys(Microprediction.get_sponsors(Microprediction.Config()))))

    plots = map(function (stream_name)
        f = loadStream(stream_name=stream_name, forecast_locations="wind")
        p = plot()

        names = filter(x -> startswith(String(x), name), colnames(f[1]))

        for name in names
            plot!(p, f[1][Symbol(name)], label=String(name), legend=:outertopright)
        end
        plot!(p, f[1][:Demand], title="Wind Energy Generation vs $(name)", label="Generation", legend=:outertopright, ylim=(-3, 3), width=3, color="black")

        p
    end, electricity_load_streams)

    big_p = plot(plots..., size=(1800, 600), layout=(size(plots, 1), 1))
    return big_p
end

function plot_streams_fuel_type_other_renewables_combined(name)
    streams = collect(filter(x -> startswith(x, "electricity-fueltype-nyiso-other_renewables"), keys(Microprediction.get_sponsors(Microprediction.Config()))))

    plots = map(function (stream_name)
        f = loadStream(stream_name=stream_name, forecast_locations="solar")
        p = plot()

        names = filter(x -> occursin(name, String(x)), colnames(f[1]))

        for name in names
            plot!(p, f[1][Symbol(name)], label=String(name), legend=:outertopright)
        end
        plot!(p, f[1][:Demand], title="Wind Energy Generation vs $(name)", label="Generation", legend=:outertopright, ylim=(-3, 3), width=2, color="black")

        p
    end, streams)

    big_p = plot(plots..., size=(1800, 600), layout=(size(plots, 1), 1))
    return big_p
end


function plot_streams_fuel_type_wind_separate(name)
    electricity_load_streams = collect(filter(x -> startswith(x, "electricity-fueltype-nyiso-wind"), keys(Microprediction.get_sponsors(Microprediction.Config()))))

    f = loadStream(stream_name=electricity_load_streams[1], forecast_locations="wind")
    names = filter(x -> startswith(String(x), name), colnames(f[1]))

    plots = map(function (name)
        p = plot()
        plot!(p, f[1][Symbol(name)], label="Forecast Wind Speed", legend=:outertopright, width=2)
        plot!(p, f[1][:Demand], title="Wind Energy Generation vs Forecast Wind Speed at Location $(name) (z-score)", label="Megawatts", legend=:outertopright, ylim=(-4, 4), width=2, color="black")
        p
    end, names[1:3])

    big_p = plot(plots..., size=(1800, 600), layout=(size(plots, 1), 1))
    return big_p
end

function plot_streams_fuel_type_other_renewables_separate(name)
    electricity_load_streams = collect(filter(x -> startswith(x, "electricity-fueltype-nyiso-other_renewables"), keys(Microprediction.get_sponsors(Microprediction.Config()))))

    f = loadStream(stream_name=electricity_load_streams[1], forecast_locations="solar")
    names = filter(x -> occursin(name, String(x)), colnames(f[1]))
    println(colnames(f[1]))
    println(names)
    plots = map(function (name)
        p = plot()
        plot!(p, f[1][Symbol(name)], label=String(name), legend=:outertopright, width=2)
        plot!(p, f[1][:Demand], title="Other Renewables Generation vs $(name) (z-score)", label="Megawatts", legend=:outertopright, ylim=(-3, 3), width=2, color="black")
        p
    end, names)

    big_p = plot(plots..., size=(1800, 800), layout=(size(plots, 1), 1))
    return big_p
end


function plot_streams_fuel_type_other_renewables_corr(full_name)
    electricity_load_streams = collect(filter(x -> startswith(x, "electricity-fueltype-nyiso-other_renewables"), keys(Microprediction.get_sponsors(Microprediction.Config()))))

    plots = map(function (stream_name)
        f = loadStream(stream_name=stream_name, zscore_features=false, forecast_locations="solar")
        p = scatter(values(f[1][:Demand]), values(f[1][Symbol(full_name)]), title=stream_name, legend=:outertopright, xlabel="generation", ylabel=full_name)
        p
    end, electricity_load_streams)

    big_p = plot(plots..., size=(1800, 600), layout=(size(plots, 1), 1))
    return big_p
end



function plot_streams_fuel_type_wind_corr()
    electricity_load_streams = collect(filter(x -> startswith(x, "electricity-fueltype-nyiso-wind"), keys(Microprediction.get_sponsors(Microprediction.Config()))))

    plots = map(function (stream_name)
        f = loadStream(stream_name=stream_name, zscore_features=false)
        p = scatter(values(f[1][:Demand]), values(f[1][:average_wind_speed_10_1]) .* 2.23, title=stream_name, legend=:outertopright, xlabel="generation", ylabel="Average Wind Speed", label="Location 1")
        scatter!(p, values(f[1][:Demand]), values(f[1][:average_wind_speed_10_2]) .* 2.23, title=stream_name, legend=:outertopright, xlabel="generation", ylabel="Average Wind speed", label="Location 2")
        p
    end, electricity_load_streams)

    big_p = plot(plots..., size=(1800, 600), layout=(size(plots, 1), 1))
    return big_p
end

function plot_streams_fuel_type_wind_raw()
    electricity_load_streams = collect(filter(x -> startswith(x, "electricity-fueltype-nyiso-wind"), keys(Microprediction.get_sponsors(Microprediction.Config()))))

    plots = map(function (stream_name)
        f = loadStream(stream_name=stream_name, zscore_features=false)
        p = plot(f[1][:average_wind_speed_10_1] .* 2.23, label="Average Wind Speed Location 1", legend=:outertopright)
    #        plot!(p, f[1][:minimum_wind_speed_10_1] .* 2.23, label="Minimum Wind Speed Location 1", legend=:outertopright)
    #        plot!(p, f[1][:maximum_wind_speed_10_1] .* 2.23, label="Maximum Wind Speed Location 1", legend=:outertopright)

        for i in 2:9
            plot!(p, f[1][Symbol("average_wind_speed_10_$(i)")] .* 2.23, label="Average Wind Speed Location $(i)", legend=:outertopright)
    #        plot!(p, f[1][Symbol("minimum_wind_speed_10_$(i)")] .* 2.23, label="Minimum Wind Speed Location 2", legend=:outertopright)
    #        plot!(p, f[1][Symbol("maximum_wind_speed_10_$(i)")] .* 2.23, label="Maximum Wind Speed Location 2", legend=:outertopright)
        end
        p
    end, electricity_load_streams)

    big_p = plot(plots..., size=(1800, 600), layout=(size(plots, 1), 1), title="Comparison of Windspeeds in Miles Per Hour for NY State Locations")
    return big_p
end


function plot_weather_variable(name)
    electricity_load_streams = collect(filter(x -> startswith(x, "electricity-fueltype-nyiso-wind"), keys(Microprediction.get_sponsors(Microprediction.Config()))))
    f = loadStream(stream_name=electricity_load_streams[1], zscore_features=false)

    names = filter(x -> startswith(String(x), name), colnames(f[1]))
    p = plot(size=(1800, 600), title="Comparision of $(name)")
    for name in names
        plot!(p, f[1][Symbol(name)] .* (contains(String(name), "wind_speed") ? 2.23 : 1), label=String(name), legend=:outertopright)
    end
    return p
end


function plot_quartiles_for_stream_diff(stream_name, lag_interval)
    stream = loadStream(stream_name=stream_name, zscore_features=true, forecast_locations="solar", lag_interval=lag_interval, load_live_data=false)

    dd = values(diff(stream[1][:Demand]))
    dist = [Normal,Cauchy, Laplace]
    fd = fit.(dist, Ref(dd))
    loglikelihood.(fd, Ref(dd))

    for d in dist
        println("$(d) $(loglikelihood(fit(d, dd), dd))")
    end

    qqplot(fd[1], dd, title="Overall Demand Diff(1) Quantile-Quantile Plot", label="Normal", dpi=150, markersize=1, markerstrokewidth=0, legend=true)
    #    qqplot!(fd[2], dd,label="Cauchy",markersize=1, markerstrokewidth=0, legend=true)
    qqplot!(fd[3], dd, label="Laplace", markersize=1, markerstrokewidth=0, legend=true)
end

function analyze_model_architectures_for_demand()
    files = filter(m -> match(r"^demand-quantile8", m) != nothing, readdir())

    used_models = []
    for filename in files
        data = deserialize(filename)

        push!(used_models, data[:model_name])
    end

    collect(countmap(used_models))
end

@enum AnalysisGraphType Area VLines CDF

function analyze_quant_diff()
    n = now(UTC)

    stream_name = "electricity-load-nyiso-overall.json"

    # Get the latest stream values.
    read_config = Microprediction.Config()
    live_lagged_values = Microprediction.get_lagged(read_config, stream_name)


    cached_nyiso_forecast = loadNYISOLoadForecasts()

    lag_interval = 12
    number_of_points = 225
    all_diffs = []

    @gif for time_offset in (60 * 24 * 1):-5:120
        t = n - Dates.Minute(time_offset)

        model_approach = CRPSRegression

        save_filename_prefix = "t1-$(model_approach)-electricity-load-nyiso-overall.json"
        model_full_filename = "$(save_filename_prefix)-lag-$(lag_interval).binary"

        saved_model = deserialize(model_full_filename)

        regressors, latest_value = regressorsForModel(
            save_filename_prefix=save_filename_prefix,
            stream_update_interval=Dates.Minute(5),
            stream_name=stream_name,
            run_start_time=t,
            number_of_points=number_of_points,
            lag_interval=lag_interval
        )

        results = runSavedModel(
                save_filename_prefix=save_filename_prefix,
                stream_name=stream_name,
                lag_interval=lag_interval,
                identity_name="foo",
                all_candidates=false,
                run_start_time=t)


        writeToJS(saved_model[:model], regressors, saved_model[:stats][:Demand], results[:points], results[:model_result], "tfjs-node")
        js_output = run(`node /Users/rusty/Development/electricity-modelling/javascript/model2.js`)
        js_cpu_out = JSON.parsefile("/tmp/model-out.json");

        p = plot(
            title="NYISO Overall Electricity Demand 1-Hour Ahead CDFs @ $(t)",
#                titlefontsize=10,
            ylims=(0, 0.006),
            xlims=(13000, 22000),
            size=(900, 900),
            xlabel="Megawatts",
            legend=:topright,
            legendfontsize=12,

        )

        plot!(p,
        kde(vec(results[:points])),
        # fillcolor=palette(:Paired_12)[(index * 2 % 12) + 1],
        # color=palette(:Paired_12)[(index * 2 % 12) + 1],
        fillalpha=0.5,
        fillrange=0,
        width=2,
       # titlefontsize=10,
        label="Julia CRPSRegression")

        js_cpu_out = convert(Array{Float32}, js_cpu_out)
        plot!(p,
        kde(vec(js_cpu_out)),
        # fillcolor=palette(:Paired_12)[(index * 2 % 12) + 1],
        # color=palette(:Paired_12)[(index * 2 % 12) + 1],
        fillalpha=0.5,
        fillrange=0,
        width=2,
        label="Javascript CRPSRegression")

        actual = values(from(live_lagged_values, t + Dates.Second(3555)))

        vline!(p, [actual[1]],
        color="red",
        width=3,
        label="Actual Value")

        quant_diffs = vec(results[:points]) .- js_cpu_out

        all_diffs = [all_diffs..., quant_diffs...]

        p2 = plot(
            title="Quantile Forecast Difference @ $(t)",
#                titlefontsize=10,
            legend=false,
            kde(vec(quant_diffs)),
            ylims=(0, 0.25),
            fillalpha=0.5,
            fillrange=0,
            width=2,

            xlims=(-500, 200),
        )

        p3 = plot(p, p2, layout=(2, 1),

        size=(1024, 1024)
        )


        p3

    end

    println(minimum(all_diffs))
    println(maximum(all_diffs))
end

function analyze_model(
    graph_type::AnalysisGraphType=Area,
    stream_name::String="electricity-fueltype-nyiso-wind.json",
    lag_interval::Number=12,
)
    n = now(UTC)


    # Get the latest stream values.
    read_config = Microprediction.Config()
    live_lagged_values = Microprediction.get_lagged(read_config, stream_name)

    cached_nyiso_forecast = loadNYISOLoadForecasts()

    all_values::Array{Float32,1} = convert(Array{Float32,1}, [])

    #        ParameterizedDistributionDiff, QuantileRegression
    model_approaches = [CRPSRegression]

    for time_offset in (60 * 24 * 1.5):(-5):60
        t = n - Dates.Minute(time_offset)


        all_results = map(model_approaches) do model_approach
            results = runSavedModel(
                save_filename_prefix="t1-$(model_approach)-$(stream_name)",
                stream_name=stream_name,
                lag_interval=lag_interval,
                identity_name="foo",
                all_candidates=false,
                run_start_time=t)


            push!(all_values, maximum(results[:points]))
            push!(all_values, minimum(results[:points]))
        end
    end

    push!(all_values, minimum(values(live_lagged_values)))
    push!(all_values, maximum(values(live_lagged_values)))


#    xlims = (minimum(all_values) * 0.9, maximum(all_values) * 1.1)
    xlims = (0, 100)
    println("Lims", xlims)


    @gif for time_offset in (60 * 24 * 1.5):-5:60
        t = n - Dates.Minute(time_offset)

        all_results = map(model_approaches) do model_approach
            results = runSavedModel(
                save_filename_prefix="t1-$(model_approach)-$(stream_name)",
                stream_name=stream_name,
                lag_interval=lag_interval,
                identity_name="foo",
                all_candidates=false,
                run_start_time=t)
            (model_approach, results)
        end

        plots = []
        actual = values(from(live_lagged_values, t + Dates.Second(3555)))
#        actual_nyiso = values(from(cached_nyiso_forecast[Symbol("nyiso-overall")], t + Dates.Second(3555)))

        if graph_type === CDF
            p = plot(
                title="$(stream_name) lag: $(lag_interval) Ahead CDFs @ $(t)",
                titlefontsize=14,
                legend=:topleft,
                xlims=xlims,
                size=(900, 600)
            )
            for (index, (approach, result)) in enumerate(all_results)
                plot!(p, vec(result[:points]), fixed_quantiles,
                fillcolor=palette(:Paired_12)[(index * 2 % 12) + 1],
                color=palette(:Paired_12)[(index * 2 % 12) + 1],
                label="$(approach) $(result[:name])")
            end
            p
        else
            p = plot(
                title="$(stream_name) lag: $(lag_interval) Ahead CDFs @ $(t)",
#                titlefontsize=10,
                legend=:topleft,
                xlims=xlims,
                size=(900, 900)
            )

            for (index, (approach, result)) in enumerate(all_results)
                if graph_type === VLines
                    p = # plot(
                vline(vec(result[:points]),
                fillcolor=palette(:Paired_12)[(index * 2 % 12) + 1],
                color=palette(:Paired_12)[(index * 2 % 12) + 1],
                fillalpha=0.25,
                fillrange=0,
                ylims=(0, 0.004),
                xlims=xlims,
                xlabel="Megawatts",
                legend=:topright,
                title="$(stream_name) 1-Hour Ahead @ $(t)",
#                titlefontsize=10,
                label="$(approach) $(result[:name])")
                elseif graph_type === Area


                    plot!(p,
                    kde(vec(result[:points])),
                    # fillcolor=palette(:Paired_12)[(index * 2 % 12) + 1],
                    # color=palette(:Paired_12)[(index * 2 % 12) + 1],
                    fillalpha=0.5,
                    fillrange=0,
                    width=2,
                    xlabel="Megawatts",
                    ylims=(0, 0.006),
                    xlims=xlims,
                    legend=:topright,
                    title="$(stream_name) lag: $(lag_interval) @ $(t)",
                    legendfontsize=12,
                   # titlefontsize=10,
                    label="$(approach)")
                end

            end

            # Now I'd like to get the true value.
            if (length(actual) > 0)
                vline!(p, [actual[1]],
            color="red",
            width=3,
            label="Actual Value")
            end


            # if (length(actual_nyiso) > 0)
            #     vline!(p, [actual_nyiso[1]],
            #         color="white",
            #         width=3,
            #         label="NYISO Public Hourly Forecast")
            # end

            push!(plots, p)

            plot(plots..., layout=(length(plots), 1),
            dpi=150, size=(900, 600))
        end
    end
end

function plot_training_history(filenames...)

    data = []
    for save_filename in filenames
        data = vcat(data, deserialize(save_filename)[:training_losses])
    end

#    test_loss_all_values = map(x -> x[:test_loss], reduce(vcat, map(x -> x[:loss_history], data)))
#    overall_loss_all_values = map(x -> x[:overall_loss], reduce(vcat, map(x -> x[:loss_history], data)))

#    y_lim = (minimum(test_loss_all_values) * 0.8, 100)

#    plotlyjs()
#    gr()

    plots = []

    # Compare all of the results.

    target_symbol = :test_total

    data = sort(data, by=x -> minimum(map(y -> y[target_symbol], x[:loss_history])))

    for row in data
        println("$(row[:name]) - $(minimum(map(y -> y[target_symbol], row[:loss_history])))")
    end

    is_first = true
    for symbol_name in [target_symbol]
        p = plot(title="$(symbol_name) from Target")
        for record in data
            v = map(x -> x[symbol_name], record[:loss_history])[30:end]
            plot!(p,
                rollmean(v, 10),
                label="$(record[:name])",
                legend=:outertopright,
                legendfontsize=6)
        end
        push!(plots, p)
    end
    final = plot(plots..., layout=(length(plots), 1), size=(1000, 1000))
    gr()
    return final
end


function plot_demands()

    stream_names = filter(x -> startswith(x, "electricity-load"), keys(Microprediction.get_sponsors(Microprediction.Config())))
    plots = []
    for stream_name in stream_names
        stream = loadStream(stream_name=stream_name, zscore_features=false, forecast_locations="solar", skip_weather=true, lag_interval=1, load_live_data=false)
        push!(plots, plot(stream[1][:Demand], title=stream_name, legend=false))
        push!(plots, histogram(diff(stream[1][:Demand], 1), title="Diff 1", bins=100))
        push!(plots, histogram(diff(stream[1][:Demand], 3), title="Diff 3", bins=100))
        push!(plots, histogram(diff(stream[1][:Demand], 12), title="Diff 12", bins=100))
    end
    return plot(plots..., layout=(length(stream_names), 4), size=(2000, 200 * length(stream_names)))
end

function plot_prediction_history()

    predictions_by_stream = Dict()
    for line in eachline("prediction-log.json")
        l = JSON.parse(line)

        if !haskey(l, "prediction_run_time")
            continue
        end
        dt = DateTime(ZonedDateTime(DateTime(l["prediction_run_time"]), tz"UTC"))
        points = l["points"]
        delay = l["delay"]

        if delay == 70
            continue
        end

        if !haskey(l, "write_key")
            write_key = "8f0fb3ce57cb67498e3790f9d64dd478"
        else
            write_key = l["write_key"]
        end

        points_directly_learned = false
        if write_key == "7e5d0f66b23def57c5f9bcee73ab45dd"
            points_directly_learned = true
        end


        if write_key != "8a8fb150f0dadaf66b918602620c30ec"
            continue
        end

        stream_name = l["stream_name"]



        if !(dt >= now(UTC) - Dates.Hour(7))
            continue
        end

        if !haskey(predictions_by_stream, stream_name)
            predictions_by_stream[stream_name] = []
        end

        push!(predictions_by_stream[stream_name], Dict(
            :identity => points_directly_learned ? "direct-points" : "parameterized-distribution",
            :points => points,
            :delay => delay,
            :dt => dt
        ))
    end

    streams = sort([
        "electricity-load-nyiso-north.json"
        "electricity-load-nyiso-centrl.json"
        "electricity-load-nyiso-hud_valley.json"
        "electricity-load-nyiso-overall.json"
        "electricity-load-nyiso-millwd.json"
        "electricity-load-nyiso-mhk_valley.json"
        "electricity-load-nyiso-nyc.json"
        "electricity-load-nyiso-capitl.json"
        "electricity-load-nyiso-genese.json"
        "electricity-load-nyiso-west.json"
        "electricity-load-nyiso-dunwod.json"
        "electricity-load-nyiso-longil.json"])


    loaded_streams = Dict()
    predictions_by_stream_by_delay = Dict()

    delay_keys = Set()

    bounds_by_stream = Dict()
    y_max = Dict()

    combined_streams = []
    for stream_name in streams
        stream = loaded_streams[stream_name] = loadStream(stream_name=stream_name,
            zscore_features=false,
            load_live_data=true,
            skip_weather=true,
            lag_interval=0)

        recent = from(stream[1], now(UTC) - Dates.Day(2))
        min_x = nothing
        max_x = nothing

        predictions_by_stream_by_delay[stream_name] = Dict()

        for row in predictions_by_stream[stream_name]
            if !haskey(predictions_by_stream_by_delay[stream_name], row[:delay])
                predictions_by_stream_by_delay[stream_name][row[:delay]] = []
            end
            push!(delay_keys, row[:delay])

            k = kde(convert(Array{Float32,1}, row[:points]))
            if !haskey(y_max, stream_name)
                y_max[stream_name] = Dict()
            end

            if !haskey(y_max[stream_name], row[:delay]) || y_max[stream_name][row[:delay]] < maximum(k.density)
                y_max[stream_name][row[:delay]] = maximum(k.density)
            end

            if min_x === nothing || min_x > minimum(k.x)
                min_x = minimum(k.x)
            end

            if max_x === nothing || max_x < maximum(k.x)
                max_x = maximum(k.x)
            end

            push!(predictions_by_stream_by_delay[stream_name][row[:delay]], row)
        end

        bounds_by_stream[stream_name] = (min_x, max_x)

        paired_predictions = collect(zip(map(x -> predictions_by_stream_by_delay[stream_name][x], sort(collect(delay_keys)))...))


        println(length(paired_predictions))

        push!(combined_streams, paired_predictions)
    end

    delay_keys = sort(collect(delay_keys))
    println(delay_keys)

    theme(:dark)

    @gif for time_index in 1:length(combined_streams[1])
        big_plots = []
        for (stream_index, stream_name) in enumerate(streams)
            stream_plots = []
            for (delay_index, delay) in enumerate(delay_keys)
                println("stream $(stream_index) delay $(delay_index) time: $(time_index)")
#                println(size(combined_streams[stream_index][delay_index]))
                row = combined_streams[stream_index][time_index][delay_index]
                expected_value = values(from(loaded_streams[stream_name][1], row[:dt] + Dates.Second(delay))[:Demand])
                k = kde(convert(Array{Float32,1}, row[:points]))

                short_stream = replace(replace(stream_name, "electricity-load-nyiso-" => ""), ".json" => "")
#                p1 = plot(k,
                p1 = vline(row[:points],
                    xlims=bounds_by_stream[stream_name],
                    ylims=(0, y_max[stream_name][delay]),
                    title="$(short_stream) $(round(row[:dt], Dates.Minute)) - $(delay) second forecast",
                    fillrange=0,
                    color=palette(:Paired_12)[stream_index],
                    # color=:lightgray, # palette(:Paired_12)[stream_index],
                    fillcolor=palette(:Paired_12)[stream_index],
                    titlefont=(13, "monaco"),
                    fillalpha=0.75,
                    legend=false,
                    label="Predicted Density")

                if length(expected_value) > 0
                    vline!(p1, [expected_value[1]], width=2, label="Actual Value", color=:white)
                end

                push!(stream_plots, p1)
            end
            stream_plot = plot(stream_plots..., layout=(length(delay_keys), 1))
            push!(big_plots, stream_plot)
        end
        plot(big_plots..., layout=(3, 4), size=(2502, 1802))
    end

#    return predictions_by_stream
end