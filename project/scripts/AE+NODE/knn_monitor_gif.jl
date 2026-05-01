using Thesis
using LuxCore
using LinearAlgebra
using Statistics
using Plots
using Printf

const AE_PATH   = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
const NODE_PATH = "data/saved_models/NODE/16/RE2500/TL1_E500_curldiv_MS_Adam_250/node_params.jld2"

# Encode the FULL simulation (training + test) on the spot. Chunked to avoid GPU OOM.
function encode_full_simdata(ae_bundle, normalizer, ae_args; chunk::Int=64, device=get_device())
    ae, ps, st = ae_bundle.ae, ae_bundle.ps, ae_bundle.st
    enc   = ae.encoder
    ps_e  = device(ps.encoder)
    st_e  = device(LuxCore.testmode(st.encoder))

    simdata = load_simdata(ae_args.full_data_path)
    preprocess_data!(simdata; verbose=false)
    N = size(simdata.u, 4)

    z_chunks = Vector{Matrix{Float32}}()
    for s in 1:chunk:N
        e = min(s + chunk - 1, N)
        idx = collect(s:e)
        x_in, _, _ = build_batch(
            EpochData(Thesis.get_data_in(simdata.u, simdata.μ₀; idx=idx)...),
            1:length(idx); normalizer=normalizer,
        )
        x_in = device(x_in)
        z, _ = enc(x_in, ps_e, st_e)
        push!(z_chunks, Array(cpu_device()(z)))
        x_in = nothing; z = nothing; GC.gc()
    end

    z_all = hcat(z_chunks...)
    t_all = simdata.time
    t_train_end = ae_args.t_training
    simdata = nothing; GC.gc()
    @info "Encoded full simulation" size_z=size(z_all) tspan=(t_all[1], t_all[end]) t_train_end
    return z_all, t_all, t_train_end
end

# auto-pick the two latent dims with the largest GT-vs-prediction drift
function pick_drift_dims(z_gt, z_pred; cutoff_idx=nothing)
    last_idx = cutoff_idx === nothing ? size(z_gt, 2) : min(size(z_gt, 2), cutoff_idx + 50)
    err = vec(sum(abs2, z_gt[:, 1:last_idx] .- z_pred[:, 1:last_idx]; dims=2))
    perm = sortperm(err; rev=true)
    return perm[1], perm[2]
end

function build_knn_demo(; AE_path::String=AE_PATH, NODE_path::String=NODE_PATH,
                          outdir::String="gifs", k::Int=5, q::Float64=1.1,
                          n_frames::Int=120, dims=nothing, chunk::Int=64)
    root = is_hpc() ? "/scratch/mfbclaeys" : ""
    AE_path   = joinpath(root, AE_path)
    NODE_path = joinpath(root, NODE_path)

    normalizer = load_normalizer(AE_path)
    ae_bundle, ae_args = load_trained_AE(AE_path)
    node, node_args = load_node(NODE_path; verbose=false)

    # encode the full simulation on the spot — z_full covers training + test
    z_full, t_full, t_train_end = encode_full_simdata(ae_bundle, normalizer, ae_args; chunk=chunk)

    # KNN reference cluster: training-portion latents, downsampled like NODE training did
    train_mask = t_full .< t_train_end
    train_idx_full = findall(train_mask)
    ds_idx = Thesis.downsample_equal(train_idx_full, node_args.downsample)
    z_train = z_full[:, ds_idx]
    knn = fit_knn_ood(z_train; k=k, q=q)
    @info "KNN OOD fit on freshly encoded training latents" threshold=knn.threshold n_train=size(z_train, 2)

    # roll out the NODE from the encoded z₀ across the whole timespan
    z0_gt = z_full[:, 1]
    z_pred = predict_array(node, z0_gt; t=t_full)
    @info "NODE rollout done" size_pred=size(z_pred)

    score_gt   = [KNN_score(knn, z_full[:, i]) for i in axes(z_full, 2)]
    score_pred = [KNN_score(knn, z_pred[:, i]) for i in axes(z_pred, 2)]

    test_start = findfirst(t -> t >= t_train_end, t_full)
    cutoff_rel = test_start === nothing ? nothing :
                 findfirst(s -> s > knn.threshold, @view score_pred[test_start:end])
    cutoff_idx = cutoff_rel === nothing ? nothing : test_start + cutoff_rel - 1
    cutoff_t   = cutoff_idx === nothing ? nothing : t_full[cutoff_idx]
    cutoff_t === nothing ? (@warn "rollout never crossed the KNN threshold") : (@info "KNN cutoff" cutoff_idx cutoff_t)

    if dims === nothing
        d1, d2 = pick_drift_dims(z_full, z_pred; cutoff_idx=cutoff_idx)
    else
        d1, d2 = dims
    end
    @info "Showing latent dims" d1 d2

    mkpath(outdir)
    static_path = joinpath(outdir, "knn_monitor_static.png")
    gif_path    = joinpath(outdir, "knn_monitor.gif")

    make_static(t_full, z_full, z_pred, score_gt, score_pred, knn.threshold,
                cutoff_t, t_train_end, d1, d2, static_path)
    make_gif(t_full, z_full, z_pred, score_gt, score_pred, knn.threshold,
             cutoff_idx, cutoff_t, t_train_end, d1, d2, gif_path, n_frames)

    return (; static_path, gif_path, cutoff_t, threshold=knn.threshold, dims=(d1, d2))
end

function make_static(t, z_gt, z_pred, s_gt, s_pred, thr, cutoff_t, t_train_end, d1, d2, outpath)
    gt_color, pred_color, cut_color, train_color = :black, :dodgerblue, :red, :green

    p1 = plot(t, z_gt[d1, :];   label="ground truth", lw=2, color=gt_color,
              ylabel="z[$d1]", legend=:topright, grid=true, ylims=(-1,1))
    plot!(p1, t, z_pred[d1, :]; label="NODE rollout", lw=2, color=pred_color)
    vline!(p1, [t_train_end]; lw=2, ls=:dot,  color=train_color, label="end of training data")
    cutoff_t !== nothing && vline!(p1, [cutoff_t]; lw=2, ls=:dash, color=cut_color, label="KNN cutoff")

    p2 = plot(t, z_gt[d2, :];   label="", lw=2, color=gt_color, ylabel="z[$d2]", grid=true, ylims=(-1,1))
    plot!(p2, t, z_pred[d2, :]; label="", lw=2, color=pred_color)
    vline!(p2, [t_train_end]; lw=2, ls=:dot,  color=train_color, label="")
    cutoff_t !== nothing && vline!(p2, [cutoff_t]; lw=2, ls=:dash, color=cut_color, label="")

    p3 = plot(t, s_gt;   label="GT KNN score", lw=2, color=gt_color,
              ylabel="KNN score", xlabel="time", grid=true, legend=:topleft)
    plot!(p3, t, s_pred; label="rollout KNN score", lw=2, color=pred_color)
    hline!(p3, [thr]; lw=2, ls=:dash, color=cut_color, label=@sprintf("threshold (q=0.99) = %.3f", thr))
    vline!(p3, [t_train_end]; lw=2, ls=:dot,  color=train_color, label="")
    cutoff_t !== nothing && vline!(p3, [cutoff_t]; lw=2, ls=:dash, color=cut_color, label="")

    p = plot(p1, p2, p3; layout=(3, 1), size=(1000, 850), dpi=200, link=:x,
             plot_title="KNN OOD monitor — NODE rollout drift",
             titlefontsize=12, left_margin=8Plots.mm, bottom_margin=4Plots.mm)
    savefig(p, outpath)
    @info "Saved $outpath"
end

function make_gif(t, z_gt, z_pred, s_gt, s_pred, thr, cutoff_idx, cutoff_t, t_train_end,
                  d1, d2, outpath, n_frames)
    gt_color, pred_color, cut_color, train_color = :black, :dodgerblue, :red, :green

    N = length(t)
    idxs = unique(round.(Int, range(2, N, length=n_frames)))

    z1_y =(-1,1)
    z2_y = (-1,1)
    s_y  = (0.0, max(maximum(s_pred), thr) * 1.1)
    xl   = (t[1], t[end])

    anim = Animation()
    for i in idxs
        cut_now   = cutoff_idx !== nothing && i >= cutoff_idx
        train_now = t[i] >= t_train_end

        p1 = plot(t[1:i], z_gt[d1, 1:i];   label="ground truth", lw=2, color=gt_color, 
                  ylabel="z[$d1]", xlims=xl, ylims=z1_y, legend=:topright, grid=true)
        plot!(p1, t[1:i], z_pred[d1, 1:i]; label="NODE rollout", lw=2, color=pred_color)
        train_now && vline!(p1, [t_train_end]; lw=2, ls=:dot,  color=train_color, label="end of training data")
        cut_now   && vline!(p1, [cutoff_t];    lw=2, ls=:dash, color=cut_color,   label="KNN cutoff")

        p2 = plot(t[1:i], z_gt[d2, 1:i];   label="", lw=2, color=gt_color,
                  ylabel="z[$d2]", xlims=xl, ylims=z2_y, grid=true)
        plot!(p2, t[1:i], z_pred[d2, 1:i]; label="", lw=2, color=pred_color)
        train_now && vline!(p2, [t_train_end]; lw=2, ls=:dot,  color=train_color, label="")
        cut_now   && vline!(p2, [cutoff_t];    lw=2, ls=:dash, color=cut_color,   label="")

        p3 = plot(t[1:i], s_gt[1:i];   label="GT score", lw=2, color=gt_color,
                  ylabel="KNN score", xlabel="time", xlims=xl, ylims=s_y, grid=true, legend=:topleft)
        plot!(p3, t[1:i], s_pred[1:i]; label="rollout score", lw=2, color=pred_color)
        hline!(p3, [thr]; lw=2, ls=:dash, color=cut_color, label="threshold")
        train_now && vline!(p3, [t_train_end]; lw=2, ls=:dot,  color=train_color, label="")
        cut_now   && vline!(p3, [cutoff_t];    lw=2, ls=:dash, color=cut_color,   label="")

        p = plot(p1, p2, p3; layout=(3, 1), size=(1000, 850), dpi=140, link=:x,
                 plot_title=@sprintf("t = %.2f", t[i]),
                 titlefontsize=12, left_margin=8Plots.mm, bottom_margin=4Plots.mm)
        frame(anim, p)
    end
    gif(anim, outpath; fps=20, show_msg=false)
    @info "Saved $outpath"
end

if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    build_knn_demo()
end
