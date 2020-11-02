struct RegularizedDense{T,LT <: Dense}
    layer::LT
    w_l1::T
    w_l2::T
end

Flux.@functor RegularizedDense

(l::RegularizedDense)(x) = l.layer(x)

penalty(l) = 0
penalty(l::RegularizedDense) =
l.w_l1 * norm(l.layer.W, 1) + l.w_l2 * norm(l.layer.W, 2)
penalty(model::Chain) = sum(penalty(layer) for layer in model)
