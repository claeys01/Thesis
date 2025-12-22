using JLD2
using Plots

includet("NODE_core.jl")

function extrapolate_node(params_path; kws...)
    
    train_node, args = load_node(params_path)
    z_train, t_train, tspan_train, z0_train = get_NODE_data(args.train_latent_path; downsample=args.downsample)
    
    z_total, t_total, tspan_total, z0_total = get_NODE_data(args.test_latent_path; downsample=args.downsample)

    # z_total, t_total, tspan_total, z0_total = get_NODE_data(args.total_latent_path; downsample=-1)
    t_test=t_total
    total_node = deepcopy(train_node)

    total_node.t = t_total
    total_node.tspan = tspan_total


    total_pred = predict_array(total_node, z0_total)

    loss = abs.(z_total - total_pred)
    # cumulative mean per-component over time
    counts = reshape(1:size(loss, 2), 1, :)
    cummean_loss = cumsum(loss, dims=2) ./ counts

    # overall mean across components (per time) and its cumulative mean
    mean_per_time = vec(mean(loss, dims=1))
    cummean_mean = cumsum(mean_per_time) ./ (1:length(mean_per_time))

    # plot cumulative mean loss
    eps_local = 1e-12
    cum_plot = plot(title="Per-component cumulative mean abs error", xlabel="t", ylabel="cum mean abs error",
                    yscale = :log10)
    for i in 1:size(cummean_loss, 1)
        plot!(cum_plot, t_total, max.(cummean_loss[i, :], eps_local), label="")
    end
    plot!(cum_plot, t_total, max.(cummean_mean, eps_local), lw=3, lc=:black, label="mean across components")

    vspan!(cum_plot, [first(t_test), last(t_test)]; fillcolor=:purple, alpha=0.1, label="test region")
    vspan!(cum_plot,[first(t_train), last(t_train)]; fillcolor=:green, alpha=0.1, label="train/val region")
    plot!(cum_plot; ylim=(0.01, 10))  # set y-limits to [-1, 1]

    display(cum_plot)


    p = plot_node_trajectory(total_node, z_total, z0_total; n_reconstruct=2)
    plot!(p; ylim=(-1, 0.5))  # set y-limits to [-1, 1]

    vspan!(p, [first(t_test), last(t_test)];
            fillcolor=:purple, alpha=0.1, label="test region")
    vspan!(p,[first(t_train), last(t_train)];
            fillcolor=:green, alpha=0.1, label="train/val region")

    display(p)
    
    nothing
end



if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    params_path = "data/NODE_models/2025-12-21_17-50-31/node_params.jld2"
    # params_path = "data/NODE_models/2025-12-21_16-05-13/node_params.jld2"
    extrapolate_node(params_path)
end

