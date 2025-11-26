using JLD2
using Plots

includet("NODE_core.jl")

function extrapolate_node(params_path; kws...)
    
    node, args = load_node(params_path)
    z_full = load(args.full_latent_path, "z")
    @load args.full_u_path data

    z_full = Float32.(cat(z_full...;dims=2))
    @show size(z_full)
    sample_period = 726:1083 # this is still hardcoded, need to alter
    println(sample_period[end])
    z_full = z_full[:,  sample_period]
    t_full = data["time"][sample_period]
    t_full .-= t_full[1]
    @show size(t_full)
    
    
    node_extr = deepcopy(node)

    extr = 44
    Δt_extr = 0.00504
    for i in 1:extr
        append!(node_extr.t, node_extr.t[end]+i*Δt_extr)
    end
    node_extr.tspan = (node_extr.t[1], node_extr.t[end])
    @show size(node.t) size(node_extr.t)
    @show node.tspan node_extr.tspan
    z, _, _, z0 = get_NODE_data(args.period_latent_path, args.period_u_path)

    pred_extr = predict_array(node_extr, z0)
    pred = predict_array(node, z0)
    
    period_pred = pred_extr[:, end]
    @show size(period_pred)
    pred_path = "data/latent_data/period_predictions/period_pred.jld2"
    JLD2.save(pred_path, "period_pred", period_pred,
                "pred_idx", sample_period[end])
    

    
    @show size(pred) size(pred_extr)

    n_reconstruct = 4
    idx_samples = round.(Int, range(1, stop=size(z, 1), length=n_reconstruct))

    pred_extr_samples = [vec(pred_extr[i, :]) for i in idx_samples] 
    pred_samples = [vec(pred[i, :]) for i in idx_samples] 
    z_full_samples = [vec(z_full[i, :]) for i in idx_samples] 

    colors = [:black, :red, :blue, :green, :purple, :orange, :yellow]
    p = plot()
    plot!(p, [node.tspan[end], node.tspan[end]], [-0.2, 0.2], color=:black, aplha=1, label="")
    plot!(p, 2 .* [node.tspan[end], node.tspan[end]], [-0.2, 0.2], color=:black, aplha=1, label="")


    for i in 1:n_reconstruct
        plot!(p, t_full,      z_full_samples[i],    label="", linestyle=:solid, lw=1.5, color=colors[i])
        plot!(p, node.t,      pred_samples[i],      label="", linestyle=:dash,  lw=3,   color=colors[i])
        plot!(p, node_extr.t, pred_extr_samples[i], label="", linestyle=:dashdot,  lw=1.5, color=colors[i])
    end
    plot!(p, [NaN], [NaN], label="data (observed)",      linestyle=:solid,   color=:black, lw=2)
    plot!(p, [NaN], [NaN], label="pred (trained)",       linestyle=:dash,    color=:black, lw=2)
    plot!(p, [NaN], [NaN], label="extr (extrapolated)",  linestyle=:dashdot, color=:black, lw=2)
    display(p)
    savefig(p, "figs/node_predictions/latent_period_pred.png")
    nothing
end



if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    params_path = "data/saved_models/NODE/16/U128_l16_tanhshrink_Tsit5/node_params.jld2"
    z_full_path = "data/latent_data/16/U_128_latent.jld2"
    # extrapolate_node(params_path)
end

