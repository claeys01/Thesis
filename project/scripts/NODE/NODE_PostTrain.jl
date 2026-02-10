using JLD2
using Plots

includet("NODE_core.jl")

# ---- Helpers ----

avg_loss_over_latent(loss::AbstractMatrix) = vec(dropdims(mean(loss, dims=1), dims=1))

function region_spans!(plt, t_train, t_test)
    vspan!(plt, [first(t_test), last(t_test)];  fillcolor=:purple, alpha=0.1, label="test region")
    vspan!(plt, [first(t_train), last(t_train)]; fillcolor=:green,  alpha=0.1, label="train region")
    return plt
end

function load_datasets(args; total_downsample::Int = -1, verbose::Bool = false)
    z_train, t_train, tspan_train, z0_train = get_NODE_data(args.train_latent_path; downsample=args.downsample, verbose=verbose)
    z_test,  t_test,  tspan_test,  z0_test  = get_NODE_data(args.test_latent_path;  downsample=args.test_downsample,  verbose=verbose)
    z_total, t_total, tspan_total, z0_total = get_NODE_data(args.total_latent_path; downsample=total_downsample, verbose=verbose)

    return (z_train=z_train, t_train=t_train, tspan_train=tspan_train, z0_train=z0_train,
            z_test=z_test,   t_test=t_test,   tspan_test=tspan_test,   z0_test=z0_test,
            z_total=z_total, t_total=t_total, tspan_total=tspan_total, z0_total=z0_total)
end

function make_nodes(train_node, t_test, tspan_test, t_total, tspan_total)
    test_node = deepcopy(train_node)
    test_node.t = t_test
    test_node.tspan = tspan_test

    total_node = deepcopy(train_node)
    total_node.t = t_total
    total_node.tspan = tspan_total

    return test_node, total_node
end


function predictions_and_losses(test_node, z0_test, z_test, total_node, z0_total, z_total)
    test_pred  = predict_array(test_node, z0_test)
    total_pred = predict_array(total_node, z0_total)

    test_loss  = abs.(z_test  .- test_pred)
    total_loss = abs.(z_total .- total_pred)

    avg_test_loss  = avg_loss_over_latent(test_loss)
    avg_total_loss = avg_loss_over_latent(total_loss)

    return test_loss, total_loss, avg_test_loss, avg_total_loss
end

function plot_trajectories(test_node, z_test, z0_test, train_node, z_train, z0_train; n_reconstruct::Int = 2, ylim=(-1, 1))
    p = plot_node_trajectory(test_node, z_test,  z0_test;  n_reconstruct=n_reconstruct)
    p = plot_node_trajectory(train_node, z_train, z0_train; n_reconstruct=n_reconstruct, plt=p, labels=false)
    plot!(p; ylim=ylim, title="NODE extrapolation", ylabel="Latent value", grid=true, minorgrid=true)
    return p
end

function plot_losses(test_node, total_node, test_loss, total_loss, avg_test_loss, avg_total_loss; yscale=:linear, ylim=(0.00000, 0.5))
    eps = 1e-12
    # ylim=()
    plt = plot(yscale=yscale, ylim=ylim)
    plot!(plt; title="NODE extrapolation MAE loss", xlabel="time", ylabel="absolute error", grid=true, minorgrid=true)

    n_lat = size(test_loss, 1)
    for i in 1:n_lat
        plot!(plt, test_node.t,  test_loss[i, :]  .+ eps; label=i == 1 ? "test losses"  : nothing, color=:red,  alpha=0.1)
        plot!(plt, total_node.t, total_loss[i, :] .+ eps; label=i == 1 ? "total losses" : nothing, color=:blue, alpha=0.1)
    end
    plot!(plt, test_node.t,  avg_test_loss  .+ eps; color=:red,  linewidth=3, label="avg test loss")
    plot!(plt, total_node.t, avg_total_loss .+ eps; color=:blue, linewidth=3, label="avg total loss")
    return plt
end

# ---- Main entry ----

function extrapolate_node(params_path; test_downsample::Int = -1, total_downsample::Int = -1, ylim=(-1, 1), verbose::Bool = false)
    train_node, args = load_node(params_path; verbose=false)

    data = load_datasets(args; total_downsample=total_downsample, verbose=verbose)
    test_node, total_node = make_nodes(train_node, data.t_test, data.tspan_test, data.t_total, data.tspan_total)
    
    ẑ_train, ẑ_test = predict_array(train_node, data.z0_train), predict_array(test_node, data.z0_test)

    test_loss, total_loss, avg_test_loss, avg_total_loss =
        predictions_and_losses(test_node, data.z0_test, data.z_test, total_node, data.z0_total, data.z_total)

    p = plot_trajectories(test_node, data.z_test, data.z0_test, train_node, data.z_train, data.z0_train; ylim=ylim)
    region_spans!(p, data.t_train, data.t_test)

    plt = plot_losses(test_node, total_node, test_loss, total_loss, avg_test_loss, avg_total_loss)
    region_spans!(plt, data.t_train, data.t_test)

    return plot(p, plt; layout=(2, 1), size=(900, 900)), (ẑ_train, ẑ_test)
    # return plot(p, plt; layout=(2, 1), size=(900, 900))

end

# ---- Script guard ----

# if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
#     # params_path = "data/NODE_models/2025-12-23_14-30-58/node_params.jld2"
#     params_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"
#     # plt, _ = extrapolate_node(params_path)
#     # display(plt)
# end

