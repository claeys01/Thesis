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
sol = Thesis.predict_array(aenode.NODE,  z0; t=simdata.time, onlysol=false)

ẑ = Array(sol[1])
@show size(ẑ)
ẑ_norms = norm.(eachcol(ẑ))



# p = plot(layout=(16, 16), size=(2000, 2000))

# subplot_idx = 1
# for i in 1:16
#     for j in 1:16
#         if i == j
#             histogram!(p[subplot_idx], ẑ[i, :], bins=30, alpha=0.6, 
#                 label="", legend=false, ticks=false)
#         else
#             scatter!(p[subplot_idx], ẑ[j, :], ẑ[i, :], markersize=1, alpha=0.3,
#                 label="", legend=false, ticks=false, color=:blue)
#         end
#         subplot_idx += 1
#     end
# end

# display(p)
# savefig(p, "figs/latent_trajectories_hist_clusters.png")
# plt = plot()
# for i in 1:16
#     plot!(plt, simdata.time, ẑ[i, :])
# end
# display(plt)

# struct MahalanobisOOD
#     μ::Vector{Float64}
#     Σinv::Matrix{Float64}
#     threshold::Float64
# end

# function fit_mahalanobis_ood(Ztrain::AbstractMatrix; q=0.99, reg=1e-6)
#     # columns = samples
#     μ = vec(mean(Ztrain, dims=2))
    
#     Σ = cov(Ztrain, dims=2)
    
#     # regularization for numerical stability
#     Σreg = Σ + reg * I(size(Σ, 1))
#     Σinv = inv(Matrix(Σreg))
    
#     # distances of training points to define threshold
#     dists = [sqrt(dot(Ztrain[:, i] - μ, Σinv * (Ztrain[:, i] - μ))) for i in axes(Ztrain, 2)]
#     @show size(dists[1])
#     threshold = quantile(dists, q)
    
#     return MahalanobisOOD(μ, Σinv, threshold)
# end

# train_OOD = fit_mahalanobis_ood(latent_data.z_train)
# @show typeof(train_OOD.threshold)

# function score(model::MahalanobisOOD, z::AbstractVector)
#     δ = z - model.μ
#     return sqrt(dot(δ, model.Σinv * δ))
# end
# using NearestNeighbors, Statistics

# struct KNNOOD
#     tree::KDTree
#     Ztrain::Matrix{Float64}
#     k::Int
#     threshold::Float64
# end

# function fit_knn_ood(Ztrain::AbstractMatrix; k=5, q=0.99)
#     tree = KDTree(Ztrain)
    
#     # score each training point against its neighbors
#     # first neighbor is itself, so use k+1 and skip first
#     train_scores = Float64[]
#     for i in axes(Ztrain, 2)
#         idxs, dists = knn(tree, Ztrain[:, i], k + 1, true)
#         # @show dists
#         push!(train_scores, mean(dists[2:end]))
#     end
    
#     threshold = quantile(train_scores, q)
#     # display(plot(train_scores))
#     return KNNOOD(tree, Matrix(Ztrain), k, threshold)
# end

# function KNN_score(model::KNNOOD, z::AbstractVector)
#     idxs, dists = knn(model.tree, z, model.k, true)
#     return mean(dists)
# end

# z_train, _, _, _ = Thesis.get_NODE_data(aenode.node_args.train_latent_path)
z_train = latent_data.z_train
@show size(z_train)
knood = fit_knn_ood(z_train)
@show knood.threshold
# knood = fit_knn_ood(aenode.node_args.)


knn_scores = []
m_scores = []
timings = Float64[]

for i in 2:length(simdata.time[2:end])
    elapsed = @elapsed z_pred = Thesis.predict_array(aenode.NODE,  z0; t=[t₀, simdata.time[i]], onlysol=true)[:, end]
    push!(timings, elapsed)
    elapsed > 0.03 && println(i, " ", elapsed)
    KNN = KNN_score(knood, z_pred)
    if KNN > knood.threshold
        z0 = latent_data.z_total[:, i]
        t₀ = simdata.time[i]
    end
    push!(knn_scores, KNN)
end

time_axis = simdata.time[2:end-1]
plt = plot(time_axis, knn_scores, label="Score over rollout time",
    xlabel="tU/L",
    ylabel="KNN score",
    ylims=(0, 2*knood.threshold), 
    xlims=(0, maximum(time_axis)))
hline!(plt, [knood.threshold], label="Threshold")
# plot!(plt, time_axis, knn_scores, )
display(plt)
# display(plot(timings))