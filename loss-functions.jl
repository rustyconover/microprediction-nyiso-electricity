LOSS_SMIDGE = Float32(0.0001)

vecnorm(x) = sum(abs2, x)

"""

    loss_total_parameterized_distribution(model, distribution, regressors, actual_values)

Compute the total loss for a parameterized distribution.
"""
function loss_total_parameterized_distribution(
    model,
    distribution,
    regressors::Array{Float32,2},
    actual_values::Array{Float32,2})::Dict{Symbol,Float32}

    model_result::Array{Float32,2} = model(regressors)
    reg_loss::Float32 = sum(vecnorm, params(model))

    mu = model_result[1, :]

    # For numeric stability ensure that the std is never zero
    # by adding a small value to it.
    std = softplus.(model_result[2, :]) .+ LOSS_SMIDGE

    likelihood_loss = -sum(map(x -> DistributionsAD.logpdf(distribution(x[1], x[2]), x[3]) + LOSS_SMIDGE, zip(mu, std, actual_values)))
    return Dict(:total => likelihood_loss + reg_loss,
                :reg_loss => reg_loss)
end

"""
    generic_loss_parameterized_distribution(model, distribution, regressors, actual_values)

Compute the loss using parameterized distributions for a batch of
examples.

"""
function generic_loss_parameterized_distribution(
    model,
    distribution,
    regressors::Array{Float32,2},
    actual_values::Array{Float32,2})::Float32

    model_result::Array{Float32,2} = model(regressors)
    reg_loss::Float32 = sum(vecnorm, params(model))

    mu = model_result[1, :]
    std = softplus.(model_result[2, :]) .+ LOSS_SMIDGE
    likelihood_loss = -sum(zip(mu, std, actual_values)) do (mu, std, y_target)
        DistributionsAD.logpdf(distribution(mu, std), y_target) + LOSS_SMIDGE
    end
    return likelihood_loss + reg_loss
end

"""
    loss_total_closest_point(model, distribution, regressors, actual_values)

Compute the total loss for a batch of examples by adding up
the squared distance to the closest produced point.
"""
function loss_total_closest_point(
    model,
    regressors::Array{Float32,2},
    actual_values::Array{Float32,2})::Dict{Symbol,Float32}

    raw_model_result::Array{Float32,2} = model(regressors)
    reg_loss::Float32 = sum(vecnorm, params(model))
    batch_size::Int32 = size(raw_model_result, 2)
    quant_count = size(raw_model_result, 1)

    # The sum of the squared distance of each closest point to the
    # actual value.


    model_result = view(cumsum(vcat(
       view(raw_model_result, 1:1, 1:batch_size),
       view(raw_model_result, 2:quant_count, 1:batch_size).^2
    ), dims=1), 2:quant_count, 1:batch_size)


    point_loss::Float32 = 0.0
    for batch_index in 1:batch_size
        point_loss += minimum((model_result[:, batch_index] .- actual_values[batch_index]).^2)
    end
#    point_loss /= convert(Float32, batch_size)

    return Dict(
        :total => point_loss + reg_loss,
        :reg_loss => reg_loss,
        :points_loss => point_loss,
    )
end

"""
    generic_loss_closest_point(model, distribution, regressors, actual_values)

Compute the total loss for a batch of examples by adding up
the squared distance to the closest produced point.

"""
function generic_loss_closest_point(model, regressors::Array{Float32,2}, actual_values::Array{Float32,2})::Float32
    raw_model_result::Array{Float32,2} = model(regressors)
    reg_loss::Float32 = sum(vecnorm, params(model))
    batch_size::Int32 = size(raw_model_result, 2)
    quant_count = size(raw_model_result, 1)

    model_result = view(cumsum(vcat(
       view(raw_model_result, 1:1, 1:batch_size),
       view(raw_model_result, 2:quant_count, 1:batch_size).^2
    ), dims=1), 2:quant_count, 1:batch_size)


    point_loss::Float32 = 0.0
    for batch_index in 1:batch_size
        point_loss += minimum((model_result[:, batch_index] .- actual_values[batch_index]).^2)
    end
#    point_loss /= convert(Float32, batch_size)

    return point_loss + reg_loss
end


"""
    loss_total_quantile(model, regressors, actual_values)

Compute the total loss for a batch of examples by adding up
the quantile loss for all of the passed quantiles.
"""
function loss_total_quantile(model, regressors::Array{Float32,2}, actual_values::Array{Float32,2})::Dict{Symbol,Float32}
    raw_model_result::Array{Float32,2} = model(regressors)
    reg_loss::Float32 = sum(vecnorm, params(model))
    batch_size::Int32 = size(raw_model_result, 2)
    quant_count = size(raw_model_result, 1)


    points_loss::Float32 = 0.0

    model_result = view(cumsum(vcat(
       view(raw_model_result, 1:1, 1:batch_size),
       view(raw_model_result, 2:quant_count, 1:batch_size).^2
    ), dims=1), 2:quant_count, 1:batch_size)

    all_errors::Array{Float32,2} = actual_values .- model_result

    fixed_data = Array{Float32,2}(undef, size(all_errors, 1), 2)
    maxes = Array{Float32,1}(undef, size(all_errors, 1))

    @inbounds for errors in eachcol(all_errors)
        broadcast!(*, fixed_data, errors, big_quantiles)
        maximum!(maxes, fixed_data)
        points_loss += sum(maxes)
    end
    points_loss /= convert(Float32, batch_size)

    return Dict(
            :total => points_loss + reg_loss,
            :reg_loss => reg_loss,
            :points_loss => points_loss,
    )
end

"""
    generic_loss_quantile(model, regressors, actual_values)

Compute the loss for a batch of examples by adding up
the quantile loss for all of the passed quantiles.
"""
function generic_loss_quantile(model, regressors::Array{Float32,2}, actual_values::Array{Float32,2})::Float32
    raw_model_result::Array{Float32,2} = model(regressors)
    quant_count = size(raw_model_result, 1)

    reg_loss::Float32 = sum(vecnorm, params(model))
    reg_loss = 0.0
    batch_size::Int32 = size(raw_model_result, 2)
    points_loss::Float32 = 0.0

    model_result = view(cumsum(vcat(
       view(raw_model_result, 1:1, 1:batch_size),
       view(raw_model_result, 2:quant_count, 1:batch_size).^2
    ), dims=1), 2:quant_count, 1:batch_size)

    @inbounds for batch_index in 1:batch_size
        errors = actual_values[:, batch_index] .- model_result[:, batch_index]
        values = errors .* big_quantiles
        points_loss += sum(maximum(values, dims=2))
    end
    points_loss /= convert(Float32, batch_size)
    return points_loss + reg_loss
end


"""
    loss_total_crps(model, regressors, actual_values)

Compute the total loss for a batch of examples by adding up
the CRPS for all of the examples.
"""
function loss_total_crps(model, regressors::Array{Float32,2}, actual_values::Array{Float32,2})::Dict{Symbol,Float32}
    raw_model_result::Array{Float32,2} = model(regressors)
    reg_loss::Float32 = sum(vecnorm, params(model))

    batch_size::Int32 = size(raw_model_result, 2)
    quant_count::Int32 = size(raw_model_result, 1)

    model_result = view(cumsum(vcat(
        view(raw_model_result, 1:1, 1:batch_size),
        view(raw_model_result, 2:quant_count, 1:batch_size).^2
    ), dims=1), 2:quant_count, 1:batch_size)

    points_loss::Float32 = crps(model_result, vec(actual_values))

    # points_loss /= convert(Float32, batch_size)

    return Dict(
            :total => points_loss + reg_loss,
            :reg_loss => reg_loss,
            :points_loss => points_loss,
    )
end

"""
    generic_loss_crps(model, regressors, actual_values)

Compute the total loss for a batch of examples by adding up
the CRPS for all of the examples.
"""
function generic_loss_crps(model, regressors::Array{Float32,2}, actual_values::Array{Float32,2})::Float32
    raw_model_result::Array{Float32,2} = model(regressors)
    reg_loss::Float32 = sum(vecnorm, params(model))

    batch_size::Int32 = size(raw_model_result, 2)
    quant_count::Int32 = size(raw_model_result, 1)

    model_result = view(cumsum(vcat(
        view(raw_model_result, 1:1, 1:batch_size),
        view(raw_model_result, 2:quant_count, 1:batch_size).^2
    ), dims=1), 2:quant_count, 1:batch_size)

    points_loss::Float32 = crps(model_result, vec(actual_values))
    # points_loss /= convert(Float32, batch_size)
    return points_loss + reg_loss
end

"""
    loss_total_crps_seperate(model, regressors, actual_values)

Compute the total loss for a batch of examples by adding up
the CRPS for each example but using seperate models for the
base value and the distribution.
"""
function loss_total_crps_seperate(model, regressors::Array{Float32,2}, actual_values::Array{Float32,2})::Dict{Symbol,Float32}
    base_result::Array{Float32,2} = model[1](regressors)
    distribution_result::Array{Float32,2} = model[2](regressors)

    reg_loss::Float32 = sum(vecnorm, params(model[1])) + sum(vecnorm, params(model[2]))
    batch_size = size(base_result, 2)
    quant_count = size(distribution_result, 1)

    model_result = view(cumsum(vcat(
        base_result,
        distribution_result,
    ), dims=1), 2:quant_count, 1:batch_size)

    points_loss::Float32 = crps(model_result, vec(actual_values))
    # points_loss /= convert(Float32, batch_size)

    return Dict(
            :total => points_loss + reg_loss,
            :reg_loss => reg_loss,
            :points_loss => points_loss,
    )
end

"""
    generic_loss_crps_seperate(model, regressors, actual_values)

Compute the total loss for a batch of examples by adding up
the CRPS for each example but using seperate models for the
base value and the distribution.
"""
function generic_loss_crps_seperate(model, regressors::Array{Float32,2}, actual_values::Array{Float32,2})::Float32
    base_result::Array{Float32,2} = model[1](regressors)
    distribution_result::Array{Float32,2} = model[2](regressors)

    reg_loss::Float32 = sum(vecnorm, params(model[1])) + sum(vecnorm, params(model[2]))
    batch_size = size(base_result, 2)
    quant_count = size(distribution_result, 1)

    model_result = view(cumsum(vcat(
        base_result,
        distribution_result,
    ), dims=1), 2:quant_count, 1:batch_size)

    points_loss::Float32 = crps(model_result, vec(actual_values))

    # points_loss /= convert(Float32, batch_size)
    return points_loss + reg_loss
end
