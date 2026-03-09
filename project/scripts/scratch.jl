using Thesis
using Statistics
using Plots
using Revise
using LinearAlgebra
# include("/home/matth/Thesis/project/scripts/data_getters/get_latent_data.jl")

# checkpoint = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
# save_path = "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000.jld2"
# train_latent, test_latent = get_latent_data(checkpoint; save_path=save_path)

# Thesis.train_NODE(NodeArgs(
#     train_latent_path = "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000_train.jld2",
#     test_latent_path =  "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000_test.jld2",
#     total_latent_path = "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000.jld2"
# ))
# x = randn(256, 256, 2, 50)

node_path = "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2"

node, node_args = load_node(node_path; verbose=true)
plt = plot_node_trajectory

data = Thesis.load_datasets(node_args; total_downsample=-1, verbose=true)
@show data.t_total
plt, _ = Thesis.extrapolate_node(node_path)
display(plt)
@show size(z_train)

mean_z_train = mean(z_train, dims=2)
max_z_train = maximum(z_train, dims=2)
min_z_train = minimum(z_train, dims=2)

function energy_timeseries(X)
    return 0.5 .* sum(abs2, X; dims=1)[1,:]
end

function energy_drift(E)
    dE = diff(E)
    return mean(abs2, dE)
end

# energy_z_train = energy_timeseries(z_train)
# energy_z_test = energy_timeseries(z_test)
# energy_z_total = energy_timeseries(z_total)


# @show size(energy_z_train)
# DE_train = energy_drift(energy_z_train)
# DE_test = energy_drift(energy_z_test)
# DE_total = energy_drift(energy_z_total)


# @show DE_train, DE_test, DE_total
# Δz = diff(z_train, dims=2)
# step_sizes = norm.(eachcol(Δz))               # should be roughly constant

# # 2. Acceleration (second derivative) — detects kinks/explosions
# Δ²z = diff(Δz, dims=2)
# acceleration = norm.(eachcol(Δ²z))

# # 3. Trajectory curvature — very sensitive to drift
# curvature = acceleration ./ (step_sizes[1:end-1] .+ 1e-8)

# # 4. Norm of the latent state — should stay within training range
# z_norms = norm.(eachcol(z_train))