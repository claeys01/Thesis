divergence_field(x̂::AbstractArray; buff::Int=1) = div_vectorized(x̂; buff=buff)
vorticity_field(x̂::AbstractArray; buff::Int=1) = curl_vectorized(x̂; buff=buff)

function _crop_buff(arr::AbstractArray, buff::Int)
    nd = ndims(arr)
    if nd == 2
        return arr[(1+buff):(end-buff), (1+buff):(end-buff)]
    elseif nd == 3
        return arr[(1+buff):(end-buff), (1+buff):(end-buff), :]
    elseif nd == 4
        return arr[(1+buff):(end-buff), (1+buff):(end-buff), :, :]
    else
        throw(ArgumentError("_crop_buff expects 2-, 3-, or 4-D array, got ndims=$nd"))
    end
end

function _scalar_mask(mask::AbstractArray)
    nd = ndims(mask)
    if nd == 4
        return @view mask[:, :, 1:1, :]
    elseif nd == 3
        return @view mask[:, :, 1:1]
    else
        return mask
    end
end

function relative_l2(x̂::AbstractArray, x::AbstractArray; mask=nothing, eps::Float32=1f-12)
    if mask === nothing
        num = sum(abs2, x̂ .- x)
        den = sum(abs2, x)
    else
        num = sum(abs2, (x̂ .- x) .* mask)
        den = sum(abs2, x .* mask)
    end
    return Float32(sqrt(num / (den + eps)))
end

function divergence_stats(x̂::AbstractArray; mask=nothing, buff::Int=1)
    d = abs.(divergence_field(x̂; buff=buff))
    if mask === nothing
        return (max=Float32(maximum(d)), mean=Float32(mean(d)))
    end
    m = _scalar_mask(mask)
    m = _crop_buff(m, buff)
    if ndims(m) == ndims(d) + 1
        m = dropdims(m; dims=3)
    end
    masked = d .* m
    n = sum(m)
    mn = n > 0f0 ? Float32(sum(masked) / n) : 0f0
    return (max=Float32(maximum(masked)), mean=mn)
end

function vorticity_relative_l2(x̂::AbstractArray, x::AbstractArray; mask=nothing, buff::Int=1, eps::Float32=1f-12)
    ω̂ = vorticity_field(x̂; buff=buff)
    ω = vorticity_field(x; buff=buff)
    m = nothing
    if mask !== nothing
        m = _scalar_mask(mask)
        m = _crop_buff(m, buff)
        if ndims(m) == ndims(ω) + 1
            m = dropdims(m; dims=3)
        end
    end
    return relative_l2(ω̂, ω; mask=m, eps=eps)
end

function wake_mask(μ₀::AbstractArray, args::LuxArgs;
                   cyl_x::Real=Float32(args.input_dim[1] / 4),
                   cyl_y::Real=Float32(args.input_dim[2] / 2),
                   D::Real=Float32(args.input_dim[2] / 8),
                   x_extent::Real=6.0,
                   y_extent::Real=2.0)
    H, W = args.input_dim[1], args.input_dim[2]
    xlo = clamp(round(Int, cyl_x), 1, H)
    xhi = clamp(round(Int, cyl_x + x_extent * D), 1, H)
    ylo = clamp(round(Int, cyl_y - y_extent * D), 1, W)
    yhi = clamp(round(Int, cyl_y + y_extent * D), 1, W)
    bbox = zeros(Float32, H, W)
    bbox[xlo:xhi, ylo:yhi] .= 1f0
    nd = ndims(μ₀)
    if nd == 4
        bbox_r = reshape(bbox, H, W, 1, 1)
    elseif nd == 3
        bbox_r = reshape(bbox, H, W, 1)
    else
        bbox_r = bbox
    end
    bbox_dev = similar(μ₀, Float32, size(bbox_r))
    copyto!(bbox_dev, bbox_r)
    return _scalar_mask(μ₀) .* bbox_dev
end

function latent_smoothness(z_traj::AbstractMatrix)
    size(z_traj, 2) >= 2 || throw(ArgumentError("latent_smoothness needs at least 2 frames"))
    diffs = z_traj[:, 2:end] .- z_traj[:, 1:(end-1)]
    norms = vec(sqrt.(sum(abs2, diffs; dims=1)))
    return (mean=Float32(mean(norms)), max=Float32(maximum(norms)))
end

function latent_pca_energy(z_traj::AbstractMatrix; k::Int=2)
    μ = mean(z_traj; dims=2)
    Z = Float64.(z_traj .- μ)
    F = svd(Z)
    s2 = F.S .^ 2
    total = sum(s2)
    total == 0 && return 0f0
    kk = min(k, length(s2))
    return Float32(sum(s2[1:kk]) / total)
end

# columns reused by sweep + evaluation drivers
const HPARAM_COLS = (:latent_dim, :λdiv, :λcurl, :λstrain, :η, :λ, :batch_size,
                     :loss, :epochs, :n_conv, :n_dense, :C_base, :normalize)

const METRIC_COLS = (:relL2_u, :relL2_v, :relL2_vec, :corr_u, :corr_v,
                     :max_div, :mean_div, :relL2_omega,
                     :relL2_u_wake, :relL2_v_wake, :relL2_omega_wake,
                     :latent_smoothness_mean, :latent_smoothness_max,
                     :latent_pca_top2_energy)

const STATUS_COLS = (:status, :reason)

# 5x baseline floor: WaterLily's own residual divergence is the noise floor below
# which no AE can do better. max(absolute_target, 5 * floor) keeps the constraint
# meaningful even if the dataset has a higher residual than expected.
function pass_fail(metrics::NamedTuple, baseline_max_div::Float32;
                   relL2_vec_max::Float32=0.10f0,
                   max_div_abs::Float32=0.05f0,
                   max_div_mult::Float32=5.0f0,
                   relL2_omega_wake_max::Float32=0.20f0)
    div_threshold = max(max_div_abs, max_div_mult * baseline_max_div)
    reasons = String[]
    metrics.relL2_vec >= relL2_vec_max && push!(reasons, "relL2_vec=$(round(metrics.relL2_vec; digits=4))≥$(relL2_vec_max)")
    metrics.max_div >= div_threshold && push!(reasons, "max_div=$(round(metrics.max_div; digits=4))≥$(round(div_threshold; digits=4))")
    metrics.relL2_omega_wake >= relL2_omega_wake_max && push!(reasons, "relL2_omega_wake=$(round(metrics.relL2_omega_wake; digits=4))≥$(relL2_omega_wake_max)")
    return isempty(reasons) ? ("PASS", "") : ("FAIL", join(reasons, "; "))
end

function compute_baseline_divergence(simdata::SimData, device, batch_size::Int)
    u = simdata.u
    μ₀ = simdata.μ₀
    T = size(u, 4)
    max_d = 0f0
    sum_d = 0.0
    n_tot = 0.0
    for i in 1:batch_size:T
        idxs = i:min(i + batch_size - 1, T)
        u_b = device(u[:, :, :, idxs])
        m_b = device(μ₀[:, :, :, idxs])
        d_field = abs.(divergence_field(u_b))
        m_scalar = dropdims(_scalar_mask(_crop_buff(m_b, 1)); dims=3)
        max_d = max(max_d, Float32(maximum(d_field .* m_scalar)))
        sum_d += Float64(sum(d_field .* m_scalar))
        n_tot += Float64(sum(m_scalar))
    end
    mean_d = n_tot > 0 ? Float32(sum_d / n_tot) : 0f0
    return (max=max_d, mean=mean_d)
end

function evaluate_checkpoint(ckpt_path::String, simdata::SimData, device, batch_size::Int)
    bundle, args = load_trained_AE(ckpt_path; device=device, return_params=true, testmode=true)
    ae, ps, st = bundle.ae, bundle.ps, bundle.st
    norm_cpu = load_normalizer(ckpt_path)
    normalizer = Normalizer(device(Float32.(norm_cpu.μ)), device(Float32.(norm_cpu.σ)), norm_cpu.method)

    u = simdata.u
    μ₀_full = simdata.μ₀
    T = size(u, 4)

    sum_diff_u = 0.0; sum_tgt_u = 0.0
    sum_diff_v = 0.0; sum_tgt_v = 0.0
    sum_diff_vec = 0.0; sum_tgt_vec = 0.0
    sum_diff_om = 0.0; sum_tgt_om = 0.0
    sum_diff_uw = 0.0; sum_tgt_uw = 0.0
    sum_diff_vw = 0.0; sum_tgt_vw = 0.0
    sum_diff_omw = 0.0; sum_tgt_omw = 0.0
    max_div = 0f0; sum_div = 0.0; n_div = 0.0
    corr_acc = zeros(Float64, 2); n_corr = 0
    z_chunks = Vector{Matrix{Float32}}()

    for i in 1:batch_size:T
        idxs = i:min(i + batch_size - 1, T)
        x_target = device(u[:, :, :, idxs])
        μ₀ = device(μ₀_full[:, :, :, idxs])
        x_in = cat(x_target, μ₀; dims=3)
        if args.normalize
            uvc = x_in[:, :, 1:2, :]
            uvc_norm, _ = normalize_batch(uvc; normalizer=normalizer)
            x_in = cat(uvc_norm, x_in[:, :, 3:4, :]; dims=3)
        end
        x̂, _ = ae(x_in, ps, st)
        x̂ = denormalize_batch(x̂, normalizer) .* μ₀
        target = x_target .* μ₀

        m_full = μ₀
        wake = wake_mask(μ₀, args)

        diff = x̂ .- target
        m_u = @view m_full[:, :, 1:1, :]
        m_v = @view m_full[:, :, 2:2, :]
        sum_diff_u += Float64(sum(abs2, (@view diff[:, :, 1:1, :]) .* m_u))
        sum_tgt_u  += Float64(sum(abs2, (@view target[:, :, 1:1, :]) .* m_u))
        sum_diff_v += Float64(sum(abs2, (@view diff[:, :, 2:2, :]) .* m_v))
        sum_tgt_v  += Float64(sum(abs2, (@view target[:, :, 2:2, :]) .* m_v))
        sum_diff_vec += Float64(sum(abs2, diff .* m_full))
        sum_tgt_vec  += Float64(sum(abs2, target .* m_full))

        ω̂ = vorticity_field(x̂)
        ω = vorticity_field(target)
        m_scalar = dropdims(_scalar_mask(_crop_buff(m_full, 1)); dims=3)
        wake_scalar = dropdims(_scalar_mask(_crop_buff(wake, 1)); dims=3)
        sum_diff_om += Float64(sum(abs2, (ω̂ .- ω) .* m_scalar))
        sum_tgt_om  += Float64(sum(abs2, ω .* m_scalar))
        sum_diff_omw += Float64(sum(abs2, (ω̂ .- ω) .* wake_scalar))
        sum_tgt_omw  += Float64(sum(abs2, ω .* wake_scalar))

        sum_diff_uw += Float64(sum(abs2, (@view diff[:, :, 1:1, :]) .* (@view wake[:, :, 1:1, :])))
        sum_tgt_uw  += Float64(sum(abs2, (@view target[:, :, 1:1, :]) .* (@view wake[:, :, 1:1, :])))
        sum_diff_vw += Float64(sum(abs2, (@view diff[:, :, 2:2, :]) .* (@view wake[:, :, 1:1, :])))
        sum_tgt_vw  += Float64(sum(abs2, (@view target[:, :, 2:2, :]) .* (@view wake[:, :, 1:1, :])))

        d = abs.(divergence_field(x̂))
        max_div = max(max_div, Float32(maximum(d .* m_scalar)))
        sum_div += Float64(sum(d .* m_scalar))
        n_div += Float64(sum(m_scalar))

        corrs = batch_corrs(target, x̂)
        corr_acc .+= Float64.(corrs)
        n_corr += 1

        z, _ = ae.encoder(x_in, ps.encoder, st.encoder)
        push!(z_chunks, Array(cpu_device()(z)))
    end

    eps = 1f-12
    relL2_u = Float32(sqrt(sum_diff_u / (sum_tgt_u + eps)))
    relL2_v = Float32(sqrt(sum_diff_v / (sum_tgt_v + eps)))
    relL2_vec = Float32(sqrt(sum_diff_vec / (sum_tgt_vec + eps)))
    relL2_omega = Float32(sqrt(sum_diff_om / (sum_tgt_om + eps)))
    relL2_u_wake = Float32(sqrt(sum_diff_uw / (sum_tgt_uw + eps)))
    relL2_v_wake = Float32(sqrt(sum_diff_vw / (sum_tgt_vw + eps)))
    relL2_omega_wake = Float32(sqrt(sum_diff_omw / (sum_tgt_omw + eps)))
    mean_div = n_div > 0 ? Float32(sum_div / n_div) : 0f0
    corr_avg = corr_acc ./ max(n_corr, 1)

    z_traj = hcat(z_chunks...)
    ls = latent_smoothness(z_traj)
    pca_e = latent_pca_energy(z_traj; k=2)

    metrics = (
        relL2_u = relL2_u, relL2_v = relL2_v, relL2_vec = relL2_vec,
        corr_u = Float32(corr_avg[1]), corr_v = Float32(corr_avg[2]),
        max_div = max_div, mean_div = mean_div, relL2_omega = relL2_omega,
        relL2_u_wake = relL2_u_wake, relL2_v_wake = relL2_v_wake,
        relL2_omega_wake = relL2_omega_wake,
        latent_smoothness_mean = ls.mean,
        latent_smoothness_max = ls.max,
        latent_pca_top2_energy = pca_e,
    )
    return metrics, args
end

function build_eval_row(name::String, metrics::NamedTuple, args::LuxArgs,
                        baseline::NamedTuple, status::String, reason::String)
    return (
        checkpoint = name,
        latent_dim = args.latent_dim, λdiv = args.λdiv, λcurl = args.λcurl, λstrain = args.λstrain,
        η = args.η, λ = args.λ, batch_size = args.batch_size, loss = args.loss,
        epochs = args.epochs, n_conv = args.n_conv, n_dense = args.n_dense, C_base = args.C_base,
        normalize = args.normalize,
        relL2_u = metrics.relL2_u, relL2_v = metrics.relL2_v, relL2_vec = metrics.relL2_vec,
        corr_u = metrics.corr_u, corr_v = metrics.corr_v,
        max_div = metrics.max_div, mean_div = metrics.mean_div,
        relL2_omega = metrics.relL2_omega,
        relL2_u_wake = metrics.relL2_u_wake, relL2_v_wake = metrics.relL2_v_wake,
        relL2_omega_wake = metrics.relL2_omega_wake,
        latent_smoothness_mean = metrics.latent_smoothness_mean,
        latent_smoothness_max = metrics.latent_smoothness_max,
        latent_pca_top2_energy = metrics.latent_pca_top2_energy,
        baseline_max_div = baseline.max,
        baseline_mean_div = baseline.mean,
        status = status, reason = reason,
    )
end
