using JLD2
using Plots

includet("NODE_core.jl")

function extrapolate_node(params_path; loss =:l2, kws...)
    # load NODE model spanning training length 0:N_train
    train_node, args = load_node(params_path; verbose=false)
    test_downsample = -1
    total_downsample = -1

    z_train, t_train, tspan_train, z0_train = get_NODE_data(args.train_latent_path; downsample=args.downsample, verbose=false)
    
    z_test, t_test, tspan_test, z0_test = get_NODE_data(args.test_latent_path; downsample=test_downsample, verbose=false)

    z_total, t_total, tspan_total, z0_total = get_NODE_data(args.total_latent_path; downsample=total_downsample, verbose=false)

    # define NODE model spanning test trajectory length N_train:N
    test_node = deepcopy(train_node)
    test_node.t = t_test
    test_node.tspan = tspan_test
    test_pred = predict_array(test_node, z0_test)
    test_loss = abs2.(z_test - test_pred)
    # test_loss = mse_loss(test_node, z_test, z0_test)

    # define NODE model spanning whole trajectory length 0:N
    total_node = deepcopy(train_node)
    total_node.t = t_total
    total_node.tspan = tspan_total
    total_pred = predict_array(total_node, z0_total)
    total_loss = abs2.(z_total .- total_pred)
    # total_loss = mse_loss(total_node, z_total, z0_total)
    
    @inline avg_loss(loss) = dropdims(mean(loss, dims=1), dims=1)
    @show test_loss
    avg_test_loss = avg_loss(test_loss)
    avg_total_loss = avg_loss(total_loss)

    p = plot_node_trajectory(test_node, z_test, z0_test; n_reconstruct=2)
    p = plot_node_trajectory(train_node, z_train, z0_train; n_reconstruct=2, plt=p, labels=false)
    # p = plot_node_trajectory(total_node, z_total, z0_total; n_reconstruct=2)

    plot!(p; 
        ylim=(-1, 1), 
        title = "NODE extrapolation",
        # xlabel = "time",
        ylabel = "Latent value",
        grid = true,
        minorgrid = true)
    
    vspan!(p, [first(t_test), last(t_test)];
            fillcolor=:purple, alpha=0.1, label="test region")
    vspan!(p,[first(t_train), last(t_train)];
            fillcolor=:green, alpha=0.1, label="train region")

    
    # compute positive y-limits safely
    eps = 1e-12
    # plt = plot(yscale=:log10, ylim=(0.000001, ymax))
    plt = plot(yscale=:linear, ylim=(0.000001, 3))
    plot!(plt;
          title = "NODE extrapolation MSE loss",
          xlabel = "time",
          ylabel = "absolute error",
          grid = true,
          minorgrid = true)
    for i in 1:args.latent_dim
        # add epsilon to avoid zeros
        plot!(plt, test_node.t, test_loss[i, :] .+ eps;
              label = i == 1 ? "test losses" : nothing,
              color = :red, alpha = 0.1)
        plot!(plt, total_node.t, total_loss[i, :] .+ eps;
              label = i == 1 ? "total losses" : nothing,
              color = :blue, alpha = 0.1)
    end
    plot!(plt, test_node.t, avg_test_loss .+ eps; color=:red, linewidth=3, label="avg test loss")
    plot!(plt, total_node.t, avg_total_loss .+ eps; color=:blue, linewidth=3, label="avg total loss")

    vspan!(plt, [first(t_test), last(t_test)];
            fillcolor=:purple, alpha=0.1, label="test region")
    vspan!(plt,[first(t_train), last(t_train)];
            fillcolor=:green, alpha=0.1, label="train region")

    combined = plot(p, plt; layout=(2, 1), size=(900, 900))
    return combined
end



if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    params_path = "data/NODE_models/2025-12-22_18-43-01/node_params.jld2"
    # params_path = "data/NODE_models/2025-12-21_16-05-13/node_params.jld2"
    extrapolate_node(params_path)
end

