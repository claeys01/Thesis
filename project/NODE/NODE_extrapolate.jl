using JLD2
using Plots

includet("NODE_core.jl")

function extrapolate_node(params_path; kws...)
    
    node, args = load_node(params_path)
    z_full = load(args.full_latent_path, "z")
    full_data = load(args.full_u_path, "data")
    period_data = load(args.period_u_path, "data")

    @show period_data["single_period_idx"]
    
    period_range = period_data["single_period_idx"]
    period_length = period_range[end] - period_range[1]

    extrapolation_range = first(period_range) : (last(period_range) + 2*period_length)

    z_full = Float32.(cat(z_full...;dims=2))
    z_extra = z_full[:,  extrapolation_range]
    t_extra = full_data["time"][extrapolation_range]
    t_extra .-= t_extra[1]
    
    node_extr = deepcopy(node)

    node_extr.t = t_extra
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
                "pred_idx", period_range[end])
        
    @show size(pred) size(pred_extr)

    n_reconstruct = 4
    idx_samples = round.(Int, range(1, stop=size(z, 1), length=n_reconstruct))

    pred_extr_samples = [vec(pred_extr[i, :]) for i in idx_samples] 
    pred_samples = [vec(pred[i, :]) for i in idx_samples] 
    z_extra_samples = [vec(z_extra[i, :]) for i in idx_samples] 

    colors = [:black, :red, :blue, :green, :purple, :orange, :yellow]
    p = plot()
    plot!(p, [node.tspan[end], node.tspan[end]], [-0.2, 0.3], color=:black, aplha=1, label="")
    plot!(p, 2 .* [node.tspan[end], node.tspan[end]], [-0.2, 0.3], color=:black, aplha=1, label="")


    for i in 1:n_reconstruct
        plot!(p, t_extra,     z_extra_samples[i],    label="", linestyle=:solid, lw=1.5, color=colors[i])
        # plot!(p, node.t,      pred_samples[i],      label="", linestyle=:dash,  lw=3,   color=colors[i])
        plot!(p, node_extr.t, pred_extr_samples[i], label="", linestyle=:dashdot,  lw=1.5, color=colors[i])
    end
    plot!(p, [NaN], [NaN], label="data (observed)",      linestyle=:solid,   color=:black, lw=2)
    # plot!(p, [NaN], [NaN], label="pred (trained)",       linestyle=:dash,    color=:black, lw=2)
    plot!(p, [NaN], [NaN], label="extr (extrapolated)",  linestyle=:dashdot, color=:black, lw=2)
    display(p)
    
    # save figure next to params_path with suffix "_extrapolation.png"
    png = ".png"
    params_dir = dirname(params_path)
    base = splitext(basename(params_path))[1]
    fig_path = joinpath(params_dir, string(base, "_extrapolation", png))
    savefig(p, fig_path)

    nothing
end



if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    params_path = "data/saved_models/NODE/16/RE2500/2e8/U128_l16_tanhshrink_Tsit5/node_params.jld2"
    extrapolate_node(params_path)
end

