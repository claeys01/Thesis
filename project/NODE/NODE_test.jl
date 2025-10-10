using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots, Revise, TimerOutputs

to = TimerOutput()

rng = Xoshiro(0)
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2]; length = datasize)
println(tsteps)
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = @timeit to "solve true ODE" Array(solve(prob_trueode, Tsit5(); saveat = tsteps))


dudt2 = Chain(x -> x .^ 3, Dense(2, 50, tanh), Dense(50, 2))
p, st = @timeit to "Lux setup" Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat = tsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    sum(abs2, ode_data .- pred)
end

# Callback function to observe training
callback = function (state, l; doplot = true)
    println("Current loss: ", l)
    # plot current prediction against data
    if doplot
      pred = predict_neuralode(state.u)
    #   plt = scatter(tsteps, ode_data[1, :]; label = "data")
    #   scatter!(plt, tsteps, pred[1, :]; label = "prediction")
        plt = plot(ode_data[1, :], ode_data[2, :]; label = "data")
        plot!(plt, pred[1, :], pred[2, :]; label = "prediction", title = "Neural ODE prediction")
      display(plot(plt))
    end
    return false
end

pinit = ComponentArray(p)
callback((; u = pinit), loss_neuralode(pinit))

# Train using the Adam optimizer
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = @timeit to "Adam training" Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.05); callback = callback, maxiters = 300)

# Retrain using the LBFGS optimizer
optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = @timeit to "LBFGS training" Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
    callback = callback, allow_f_increases = false)

callback((; u = result_neuralode2.u), loss_neuralode(result_neuralode2.u); doplot = true)

show(to)