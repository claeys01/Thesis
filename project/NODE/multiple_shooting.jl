using ComponentArrays, Lux, DiffEqFlux, Optimization, OptimizationPolyalgorithms,
      OrdinaryDiffEq, Plots
using DiffEqFlux: group_ranges

using Random
rng = Xoshiro(0)

# Define initial conditions and time steps
datasize = 30
u0 = Float32[2.0, 0.0]
tspan = (0.0f0, 5.0f0)
tsteps = range(tspan[1], tspan[2]; length = datasize)

# Get the data
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(); saveat = tsteps))

# Define the Neural Network
nn = Chain(x -> x .^ 3, Dense(2, 16, tanh), Dense(16, 2))
p_init, st = Lux.setup(rng, nn)

ps = ComponentArray(p_init)
pd, pax = getdata(ps), getaxes(ps)

neuralode = NeuralODE(nn, tspan, Tsit5(); saveat = tsteps)
prob_node = ODEProblem((u, p, t) -> nn(u, p, st)[1], u0, tspan, ComponentArray(p_init))
# @show typeof(prob_node)
# Define parameters for Multiple Shooting
group_size = 3
continuity_term = 200

function loss_function(data, pred)
    return sum(abs2, data - pred)
end

l1, preds = multiple_shoot(ps, ode_data, tsteps, prob_node, loss_function,
    Tsit5(), group_size; continuity_term)

function loss_multiple_shooting(p)
    ps = ComponentArray(p, pax)

    loss, currpred = multiple_shoot(ps, ode_data, tsteps, prob_node, loss_function,
        Tsit5(), group_size; continuity_term)
    global preds = currpred
    return loss
end

function plot_multiple_shoot(plt, preds, group_size)
    step = group_size - 1
    ranges = group_ranges(datasize, group_size)

    for (i, rg) in enumerate(ranges)
        plot!(plt, tsteps[rg], preds[i][1, :]; markershape = :circle, label = "Group $(i)")
    end
end

anim = Plots.Animation()
iter = 0
function callback(state, l; doplot = true, prob_node = prob_node)
    display(l)
    global iter
    iter += 1
    if doplot && iter % 1 == 0
        # plot the original data
        plt = scatter(tsteps, ode_data[1, :]; label = "Data")
        # plot the different predictions for individual shoot
        l1, preds = multiple_shoot(
            ComponentArray(state.u, pax), ode_data, tsteps, prob_node, loss_function,
            Tsit5(), group_size; continuity_term)
        plot_multiple_shoot(plt, preds, group_size)

        frame(anim)
        display(plot(plt))
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pd)
res_ms = Optimization.solve(optprob, PolyOpt(); callback = callback, maxiters = 300)
gif(anim, "multiple_shooting.gif"; fps = 15)