# The goal here is to serialize an entire suite of models.
# the question is how to serialize the individual models
# should they be seperate files or one big file.

function writeToJSFull(model_filename, output_filename)

    # regressors, latest_value = regressorsForModel(
    #         save_filename_prefix="t1-CRPSRegression-electricity-load-nyiso-overall.json",
    #         stream_update_interval=Dates.Minute(5),
    #         stream_name="electricity-load-nyiso-overall.json",
    #         run_start_time=DateTime("2020-12-26T17:55:00"),
    #         number_of_points=225,
    #         lag_interval=1
    #     );


    # julia_results = runSavedModel(
    #             save_filename_prefix="t1-CRPSRegression-electricity-load-nyiso-overall.json",
    #             stream_name="electricity-load-nyiso-overall.json",
    #             lag_interval=1,
    #             identity_name="foo",
    #             all_candidates=false,
    #             run_start_time=DateTime("2020-12-26T17:55:00"))

    model_info = deserialize(model_filename)

    open("/Users/rusty/Development/electricity-modelling/javascript/src/$(output_filename).ts", "w") do out
        write(out, """
        import * as tf from '@tensorflow/tfjs';
        import moment from 'moment';
        import * as qs from 'querystring';
        import * as microprediction from 'microprediction';

        const bent = require('bent');
        const getJSON = bent('json');
        const reverse_z = (v: number, data: number[]) => (v * data[1]) + data[0];
        const zscore = (v: number, data: number[]) => (v - data[0]) / data[1];

        """);

#        write(out, "require('@tensorflow/tfjs');\n");
#        write(out, "tf.setBackend('wasm').then(() => main());");
#        write(out, "main('2020-12-26T18:00:00');\n");

        write(out, """async function main(forecast_time: string) {
const ft = moment.utc(forecast_time);
const stream_start = moment.utc($(JSON.json(model_info[:stream_start])));
""");


        write(out, "const model = tf.sequential();\n");
        function writeTensor(t)
            if size(t, 2) > 1
                nice_values = []
                for col in eachcol(t)
                    for v in col
                        push!(nice_values, v)
                    end
                end
                values = JSON.json(nice_values)

#                values = JSON.json(map(row -> row, eachcol(t))...)
                return "tf.tensor(new Float32Array($(values)), [$(size(t, 2)), $(size(t, 1))])"
            else
                values = JSON.json(t)
                return "tf.tensor(new Float32Array($(values)), [$(size(t, 1))])"
            end
        end

        for (index, layer) in enumerate(model_info[:model])

            activation = "$(layer.σ)"
            if activation == "identity"
                activation = "linear";
            end

            if index === 1
                write(out, "model.add(tf.layers.dense({ activation: '$(activation)', units: $(size(layer.W, 1)), inputShape: [1, $(size(layer.W, 2))], weights: [$(writeTensor(layer.W)), $(writeTensor(layer.b))]}));\n")
            else
                write(out, "model.add(tf.layers.dense({ activation: '$(activation)', units: $(size(layer.W, 1)), weights: [$(writeTensor(layer.W)), $(writeTensor(layer.b))]}));\n")
            end
        end

        # Write out the regressor names.
        write(out, "const model_regressor_names = $(JSON.json(map(x -> String(x), model_info[:regressors])));\n");

        # The regressor table can be smaller and only include the regressors used
        # by this model, so cut it down.

        # Need to map the regressors to how to request the data from the feature server

        # Need to map all of the forecast locations info the H3 index
        # that will be passed to the feature server.

        locations = missing
        if model_info[:forecast_locations] === "city"
            locations = city_weather_stations
        elseif model_info[:forecast_location] === "solar"
            locations = solar_generation_sites
        elseif model_info[:forecast_locations] === "wind"
            locations = wind_generation_sites
        else
            throw(ErrorException("Unknown forecast locations specified"));
        end

        locations_to_h3 = Dict(map(locations) do location
            h3_index = location_to_h3_index(location)
            return Pair(lowercase(location[1]), string(h3_index, base=16))
        end)

        write(out, "const locations_to_h3 = $(JSON.json(locations_to_h3));\n")


        needed_weather = Set()
        # What forecast products are needed for this model?
        for regressor_name in filter(x -> startswith(String(x), "hrrr_"), model_info[:regressors])
            s = String(regressor_name)
            println(s)
            println(typeof(s))
            s = replace(s, "hrrr_" => "")

            for location in locations
                s = replace(s, "_$(get_location_name(location))" => "");
            end

            push!(needed_weather, s)
        end

        write(out, "const needed_weather_features = $(JSON.json(needed_weather));\n");

        used_regressor_stats = filter(x -> x.first in model_info[:regressors] || x.first == :Demand, model_info[:stats]);

        write(out, "const regressor_zscore_stats: {
            [variable_name: string]: [number, number]
        } = $(JSON.json(used_regressor_stats));\n");

        # To generate the regressor data for sin_228 and the other periodic functions
        # The start time of the data set needs to be expressed.

#        const julia_regressors = $(JSON.json(regressors));

#        console.log(
#            "Regressor diffs"
#        );
#        for (let j = 0; j < zscored_regressors.length; j++) {
#            console.log(model_regressor_names[j], zscored_regressors[j] - julia_regressors[j]);
#        }
# const julia_points = $(JSON.json(julia_results[:points]));

#        console.log("Results diff");
#        for (let j = 0; j < points.length; j++) {
#            console.log(julia_points[0][j] - points[j]);
#        }



        write(out, """
        const url_parameters = qs.encode({
            time: forecast_time,
            products: needed_weather_features.join(","),
            h3_indexes: Object.values(locations_to_h3).join(","),
        });

        // console.log(url_parameters);

        const weather_data: {
            [product_name: string]: {
                [h3_index: string]: number
            }
        } = await getJSON(`https://api.ionized.cloud/weather?` + url_parameters);

        // console.log(weather_data);

        const flat_weather: { [regressor_name: string]: number } = {};
        // Flatten the weather data into name and values.

        const inverse_locations: { [h3_index: string]: string } = {};
        for (const [location_name, h3_index] of Object.entries(locations_to_h3)) {
            inverse_locations[h3_index] = location_name.replace(/ /g, "_");
        }
        for (const [product_name, values] of Object.entries(weather_data)) {
            for (const [h3_index, value] of Object.entries(values)) {
                const name = [`hrrr`, product_name, inverse_locations[h3_index]].join("_");
                flat_weather[name] = value;
            }
        }
        // Start to build up the regressors.
        const periodic_ticks = moment.duration(ft.diff(moment(stream_start))).as('minutes') / 5;

        const periodic_regressors: { [name: string]: number } = {
            sin_288: Math.sin(2 * Math.PI * periodic_ticks / 288),
            cos_288: Math.cos(2 * Math.PI * periodic_ticks / 288),
            sin_2016: Math.sin(2 * Math.PI * periodic_ticks / 2016),
            cos_2016: Math.cos(2 * Math.PI * periodic_ticks / 2016),
        };


        const read_config = await microprediction.MicroReaderConfig.create({});
        const reader = new microprediction.MicroReader(read_config);

        const final_regressors: number[] = []
        for (const regressor_name of model_regressor_names) {
            if (regressor_name.match(/^hrrr_/)) {
                const v = flat_weather[regressor_name];
                if (v == null) {
                    const known_weather_keys = Object.keys(flat_weather)
                    known_weather_keys.sort();
                    // console.log(known_weather_keys.join(","));
                    throw new Error(`Failed to find weather regressor: ` + regressor_name);
                }
                final_regressors.push(v);
            } else if (periodic_regressors[regressor_name] != null) {
                final_regressors.push(periodic_regressors[regressor_name]);
            } else if (regressor_name === 'Demand_lag') {
                const values = await reader.get_lagged($(JSON.json(model_info[:stream])));

                const first_smaller_index = values.findIndex(v => v[0] < (ft.valueOf() / 1000));
                final_regressors.push(values[first_smaller_index+1][1]);
                // console.log("Demand value:", values[first_smaller_index+1][1]);
            } else if (regressor_name.match(/^nyiso-/)) {
                const v = weather_data["nyiso"][regressor_name];
                if (v == null) {
                    throw new Error(`Failed to find nyiso regressor:` + regressor_name);
                }
                final_regressors.push(v);
            } else {
                throw new Error("Unhandled regressor:" + regressor_name);
            }
        }

        const zscored_regressors = model_regressor_names.map((name, idx) => {
            // Don't zscore the periodic regressors.
            if (!name.match(/^(sin|cos)_/)) {
                return zscore(final_regressors[idx], regressor_zscore_stats[name]);
            } else {
                return final_regressors[idx];
            }
        });


        const result = model.predict(tf.tensor(zscored_regressors, [1, 1, zscored_regressors.length], 'float32'));
        if(Array.isArray(result)) {
            throw new Error("Prediction resulted in an unexpected array");
        }
        const full_result = result.dataSync() as Float32Array;

        // Now that the final results are there, we need to process them.
        for (let i = 1; i < full_result.length; i++) {
            full_result[i] = full_result[i] ** 2;
        }

        full_result[1] += full_result[0];
        for (let i = 2; i < full_result.length; i++) {
            full_result[i] += full_result[i - 1];
        }

        const points = full_result.slice(1)
            .map(v => reverse_z(v, regressor_zscore_stats["Demand"]));

        return points;

        """)

        # Here is where we call the feature vector creator.


        write(out, "}\n");


    end
end


function writeToJS(model, regressors, zscores, final_points, final_raw, module_name)
    open("/Users/rusty/Development/electricity-modelling/javascript/model2.js.tmp", "w") do out
        write(out, "const tf = require('@tensorflow/tfjs');\n");
#        write(out, "require('@tensorflow/tfjs');\n");
#        write(out, "tf.setBackend('wasm').then(() => main());");
        write(out, "main();\n");

        write(out, "function main() {\n");
        write(out, "model = tf.sequential();\n");
        function writeTensor(t)
            if size(t, 2) > 1
                nice_values = []
                for col in eachcol(t)
                    for v in col
                        push!(nice_values, v)
                    end
                end
                values = JSON.json(nice_values)

#                values = JSON.json(map(row -> row, eachcol(t))...)
                return "tf.tensor(new Float32Array($(values)), [$(size(t, 2)), $(size(t, 1))])"
            else
                values = JSON.json(t)
                return "tf.tensor(new Float32Array($(values)), [$(size(t, 1))])"
            end
        end

        for (index, layer) in enumerate(model)

            activation = "$(layer.σ)"
            if activation == "identity"
                activation = "linear";
            end

            if index === 1
                write(out, "model.add(tf.layers.dense({ activation: '$(activation)', units: $(size(layer.W, 1)), inputShape: [1, $(size(layer.W, 2))], weights: [$(writeTensor(layer.W)), $(writeTensor(layer.b))]}));\n")
            else
                write(out, "model.add(tf.layers.dense({ activation: '$(activation)', units: $(size(layer.W, 1)), weights: [$(writeTensor(layer.W)), $(writeTensor(layer.b))]}));\n")
            end
        end

        # Write the regressors.asdf
        write(out, "regressors = tf.tensor(new Float32Array($(JSON.json(regressors))), [1, 1, $(size(regressors, 1))], 'float32');\n")

        write(out, "model_result = tf.tensor(new Float32Array($(JSON.json(final_raw))), [1, 1, $(size(final_raw, 1))], 'float32');\n")

        write(out, "result = model.predict(regressors).dataSync();\n")

        write(out, "
        let current = result[0];
        let final_result = [];
        let zscore_demand = $(JSON.json(zscores));
        let julia_points = $(JSON.json(final_points));
        const reverse_z = (v) => (v * zscore_demand[1]) + zscore_demand[0]
        for(let i =1; i < result.length; i++) {
            current = current + result[i]**2;
            final_result.push(reverse_z(current));
        }
        const fs = require('fs');
        fs.writeFileSync(`/tmp/model-out.json`, JSON.stringify(final_result));
        console.log(`Finished`);
        ");
        write(out, "}\n");

        # Lets write the reverse zscore.


    end
    mv("/Users/rusty/Development/electricity-modelling/javascript/model2.js.tmp",
    "/Users/rusty/Development/electricity-modelling/javascript/model2.js", force=true)
end