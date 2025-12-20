using JLD2
using Revise
using DiffEqFlux: group_ranges
using Random
using OptimizationPolyalgorithms
rng = Xoshiro(0)

includet("../AE/Lux_AE.jl")        # load AE utilities (for latent data types)
includet("NODE_core.jl")           # NODE helpers (args, data loading, etc.)

# Configure NODE training and load latent training data
args = NodeArgs()
z, t, tspan, z0 = get_NODE_data(args.train_latent_path; downsample=args.downsample)
@show size(z)                      # z is (latent_dim, n_timepoints)

# Define the NODE right-hand side network (d/dt z = nn(z))
nn = Chain(
    Dense(args.latent_dim, args.dense_mult * args.latent_dim, args.activation),
    Dense(args.dense_mult * args.latent_dim, args.latent_dim)
)

# Initialize network parameters and state
p_init, st = Lux.setup(rng, nn)

# Wrap parameters for Optimization.jl and keep axes for reconstruction later
ps = ComponentArray(p_init)
pd, pax = getdata(ps), getaxes(ps)

# Build the NeuralODE and the equivalent ODEProblem for multiple shooting
neuralode = NeuralODE(nn, tspan, Tsit5(); saveat = t)
prob_node = ODEProblem((u, p, t) -> nn(u, p, st)[1], z0, tspan, ComponentArray(p_init))

# Simple squared-error loss between data and prediction
function loss_function(data, pred)
    return sum(abs2, data - pred)
end

# One pass of multiple shooting to inspect initial loss/predictions
# l1, preds = multiple_shoot(ps, z, t, prob_node, loss_function,
#     Tsit5(), args.group_size; args.continuity_term)

# Optimization objective: runs multiple shooting for given parameters p
function loss_multiple_shooting(p)
    ps = ComponentArray(p, pax)
    loss, currpred = multiple_shoot(ps, z, t, prob_node, loss_function,
        Tsit5(), args.group_size; args.continuity_term)
    global preds = currpred        # keep last group predictions for plotting
    return loss
end

function plot_multiple_shoot(plt, preds, group_size)
    # preds is a Vector of matrices, one per group
    step = args.group_size - 1
    ranges = group_ranges(size(z, 2), args.group_size)
    for (i, rg) in enumerate(ranges)
        plot!(plt, t[rg], preds[i][1, :]; label = "Group $(i)")
    end
end

# Optional visualization during training (disabled by default for speed)
anim = Plots.Animation()
iter = 0
function callback(state, l; doplot = false, prob_node = prob_node)
    # lightweight logging
    global iter
    iter += 1
    @info "$(iter): $l"
    if doplot && iter % 1 == 0
        # plot original series and per-group predictions
        plt = scatter(t, z[1, :]; label = "Data")
        l1, preds = multiple_shoot(
            ComponentArray(state.u, pax), z, t, prob_node, loss_function,
            Tsit5(), args.group_size; args.continuity_term)
        plot_multiple_shoot(plt, preds, args.group_size)
        frame(anim)
        display(plot(plt))
    end
    return false                   # continue optimization
end

# Set up and run optimization (PolyOpt = polyalgorithm without LR)
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pd)
res_ms = Optimization.solve(optprob, PolyOpt(); callback = callback, maxiters = 300)
gif(anim, "multiple_shooting.gif"; fps = 15)

# Extract trained parameters with the original axes for nn
p_trained = ComponentArray(res_ms.u, pax)

# Predict trajectory over t using trained parameters
neuralode = NeuralODE(nn, tspan, Tsit5(); saveat = t)
sol = neuralode(z0, p_trained, st)
pred = Array(isa(sol, Tuple) ? sol[1] : sol)
@info "Prediction size $(size(pred))"

# Report final loss against latent training data
loss = loss_function(z, pred)
@show loss