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

