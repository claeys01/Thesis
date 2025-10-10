using JLD2 

using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots, Revise, TimerOutputs

@load "matth/data/RHS_shedding_data.jld2" RHS_data

# Select indices corresponding to time 50 to 75
time_indices = findall(t -> t ≥ 50 && t ≤ 75, RHS_data["time"])
selected_indices = time_indices

# Downsample to ~50 entries
n_samples = 50
downsampled_indices = round.(Int, range(1, length(selected_indices), length=n_samples))
final_indices = selected_indices[downsampled_indices]

# Downsample all relevant entries in RHS_data
RHS_data["time"] = RHS_data["time"][final_indices]
RHS_data["Δt"] = RHS_data["Δt"][final_indices]
RHS_data["RHS"] = RHS_data["RHS"][final_indices]

println("Downsampled to ", length(RHS_data["time"]), " time steps.")

RHS_data["flattened"] = [vec(r) for r in RHS_data["RHS"]]


input_dim = size(RHS_data["flattened"][1])[1]
println("Input dim: ", input_dim, " time series length: ", length(RHS_data["time"]))

tspan = (RHS_data["time"][1], RHS_data["time"][end])
tsteps = range(tspan[1], tspan[2]; length=length(RHS_data["time"]))

dudt = Chain(Dense(input_dim, 50, tanh), Dense(50, input_dim))
p, st = Lux.setup(Xoshiro(0), dudt)

RHS_neuralode = NeuralODE(dudt, tspan, Tsit5(); saveat=tsteps)


function predict_neuralode(p)
    Array(RHS_neuralode(RHS_data["flattened"][1], p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    target = hcat(RHS_data["flattened"]...)  # shape: 100360 × 1846
    sum(abs2, target .- pred)
end

callback = function (l)
    println("Current loss: ", l)
    return false
end

pinit = ComponentArray(p)
callback(loss_neuralode(pinit))

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.01); callback = callback, maxiters = 100)