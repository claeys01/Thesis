using JLD2

# includet("prelatent_NODE.jl")
includet()

function extrapolate_node()
    args = NodeArgs()

    saved = JLD2.load(params_path)
    @info "Loaded keys from $params_path: $(collect(keys(saved)))"
    p_loaded = haskey(saved, "optimized_params") ? saved["optimized_params"] : first(values(saved))

end



extrapolate_node()

