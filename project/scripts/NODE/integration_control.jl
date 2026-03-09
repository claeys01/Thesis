using Thesis
using Plots
using LinearAlgebra
using Statistics
using KernelDensity

node_path = "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)

latent_data = Thesis.load_datasets(aenode.node_args)

simdata = load_simdata(aenode.ae_args.full_data_path)

t₀ = simdata.time[1]

rollout_range = collect(1:3000)
time = simdata.time[rollout_range]
z0 = latent_data.z_total[:, rollout_range[1]]
sol = Thesis.predict_array(aenode.NODE,  z0; t=time, onlysol=false)

ẑ = Array(sol[1])
@show size(ẑ)
ẑ_norms = norm.(eachcol(ẑ))

plt = plot()
for i in 1:16
    plot!(plt, time, ẑ[i, :])
end

struct MahalanobisOOD
    μ::Vector{Float64}
    Σinv::Matrix{Float64}
    threshold::Float64
end

function fit_mahalanobis_ood(Ztrain::AbstractMatrix; q=0.99, reg=1e-6)
    # columns = samples
    μ = vec(mean(Ztrain, dims=2))
    
    X = Ztrain .- μ
    # Σ = (X * X') / (size(Ztrain, 2) - 1)
    Σ = cov(Ztrain, dims=2)
    
    # regularization for numerical stability
    Σreg = Σ + reg * I(size(Σ, 1))
    Σinv = inv(Matrix(Σreg))
    
    # distances of training points to define threshold
    dists = [sqrt(dot(Ztrain[:, i] - μ, Σinv * (Ztrain[:, i] - μ))) for i in axes(Ztrain, 2)]
    threshold = quantile(dists, q)
    
    return MahalanobisOOD(μ, Σinv, threshold)
end

train_OOD = fit_mahalanobis_ood(latent_data.z_train)
@show typeof(train_OOD.threshold)

function score(model::MahalanobisOOD, z::AbstractVector)
    δ = z - model.μ
    return sqrt(dot(δ, model.Σinv * δ))
end

scores = []
for i in 2:length(simdata.time[2:end])
    z_pred = Thesis.predict_array(aenode.NODE,  z0; t=[t₀, simdata.time[i]], onlysol=true)[:, end]

    temp = score(train_OOD, z_pred)
    push!(scores, temp)
end

@show size(scores)
time_axis = simdata.time[2:end-1]
plt = plot(time_axis, scores, label="Score over rollout time",
    xlabel="tU/L",
    ylabel="Mahalanobis score",
    ylims=(0, 2*train_OOD.threshold), 
    xlims=(0, maximum(time_axis)))
hline!(plt, [train_OOD.threshold], label="Threshold")
