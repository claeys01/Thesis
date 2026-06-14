using JLD2
using Statistics
using Plots
using Printf
using Thesis
using LaTeXStrings
using FFTW
using DSP
using Random

basedir = "data/saved_models/inline_runs_hpc/RE250/2026-06-14_13-10"
model1 = "$basedir/AE_Jun14-1313__E500_HW256x256_C4to2_nc5_nd1_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0"
model2 = "$basedir/AE_Jun14-1329__E100_HW256x256_C4to2_nc5_nd1_z16_C8_lr0p0002_wd0p0009_bs16_NY_LL1_Tl0p0"
model3 = "$basedir/AE_Jun14-1337__E100_HW256x256_C4to2_nc5_nd1_z16_C8_lr0p0002_wd0p0009_bs16_NY_LL1_Tl0p0"

models = [model1, model2, model3]
model_labels = ["initial", "retrain 1", "retrain 2"]
p = Thesis.plot_losses(joinpath(model1, "loss_trajectory.jld2"), joinpath(model1, "checkpoint.jld2"))


hs_path = joinpath(basedir, "hybrid_state.jld2")
@load hs_path res sim_meanflow ref_meanflow params mode_log n_integrs AE_path node_path savedir

plt_forces = plot_forces_comparison(res, 100; mode_log = mode_log)
plt_forces = plot!(plt_forces, ylims=(-2, 1.5))
display(plt_forces)
savefig(plt_forces, joinpath(basedir, "plt_forces.pdf"))

function hybrid_reference(hyb, ref)
    hyb = plot!(hyb, colorbar=false)
    ref = plot(ref, ylabel="", yformatter=_ -> "")
    plt_comb = plot(hyb, ref;
        layout=grid(1, 2; widths=[0.4425, 0.5575]),
        link=:y, legend=false, size=(700, 350), dpi=600, grid=false)
    return plt_comb
end

meanplot, meanplot_panels = plot_meanflow_comparison(sim_meanflow, ref_meanflow;
    savedir=joinpath(basedir, "meanflow_panels"))
# display(meanplot)
u_ref, u_hybrid, _, v_ref, v_hybrid, _ = meanplot_panels
# u_hybrid = plot!(u_hybrid[2], colorbar=false)
# display(u_hybrid)
plt_comb = hybrid_reference(u_hybrid[2], u_ref[2])

rst_plot, rst_panels = plot_rst_comparison(sim_meanflow, ref_meanflow; style=:filled,
    savedir=joinpath(basedir, "rst_panels"))
display(rst_plot)
# display(p)
# # p_rec = Thesis.visualize_reconstructions(joinpath(model1, "checkpoint.jld2"))

# checkpoint_path = joinpath(model1, "checkpoint.jld2")
# ae_bundle, args = load_trained_AE(checkpoint_path; return_params=true)
# args.full_data_path = "data/saved_models/inline_runs_hpc/RE250/2026-06-14_13-10/U_inline_RE250.jld2"
# ae, ps, st = ae_bundle.ae, ae_bundle.ps, ae_bundle.st
# normalizer = load_normalizer(checkpoint_path)

# args.seed > 0 && Random.seed!(args.seed)
# simdata = load_simdata(args.full_data_path)

# span = length(simdata.time)
# id = randperm(span)[1]

# u_unprocessed = simdata.u[:, :, :, id]
# @show simdata.time[id]
# preprocess_data!(simdata; verbose=false)


# μ₀ = simdata.μ₀[:, :, :, id:id]
# x = simdata.u[:, :, :, id:id]
# simdata = nothing
# GC.gc()
# ##

# x_target, _ = normalize_batch(x; normalizer=normalizer)
# x_in = cat(x_target, μ₀; dims=3)

# x̂_norm, _ = ae(x_in, ps, st)


# # Denormalize on device, then move to CPU
# x̂ = denormalize_batch(x̂_norm, normalizer)
# x_target = denormalize_batch(x_target, normalizer)

# sim = circle_shedding_biot(Re=250, perturb=false)
# Thesis.apply_prediction!(sim, x̂[:, :, :, 1], 0.35f0, 1, impose_biot=true)

# û = sim.flow.u .* sim.flow.μ₀
# plt_curl_in  = Thesis.curl_contour(u_unprocessed; colorbar=false)

# # Left panel keeps its y-axis; right panel drops the y-axis (shared) and carries
# # the single colorbar. Negative margins pull them together, like the meanflow plot.
# plot!(plt_curl_in; right_margin=-1Plots.mm)
# plt_curl_out = Thesis.curl_contour(û; colorbar=true)

# plot!(plt_curl_out; left_margin=-1Plots.mm, ylabel="", yformatter=_ -> "",
#       title="", link=:y)

# curl_comp = plot(plt_curl_in, plt_curl_out;
#     layout=grid(1, 2; widths=[0.4425, 0.5575]),
#     link=:y, legend=false, size=(700, 350), dpi=600, grid=false, bottom_margin=3Plots.mm)

# display(curl_comp)
# savefig(curl_comp, joinpath(model1, "curl_comp.png"))

# plots = []
# diff_plots = []
# for ch in 1:2
#     mat_in = x_target[:, :, ch, 1] .* μ₀[:, :, ch, 1]
#     mat_out = x̂[:, :, ch, 1] .* μ₀[:, :, ch, 1]

#     # clims = Thesis.sym_clims(mat_in, mat_out)   
#     min = abs(minimum((minimum(mat_in), minimum(mat_out))))
#     max =  maximum((maximum(mat_in), maximum(mat_out)))
    
#     μ = mean(mat_in)
#     clims = (μ - min, μ + max)

#     σ = std(mat_in)*2
#     clim = (μ - σ, μ + σ)
#     cmap = cgrad(:RdBu_11, rev=true)

#     img_in  = Thesis.meanflow_contour(mat_in;  clims=clim, cmap=cmap, levels=12, linewidth= 0.1,colorbar=false)
#     img_out = Thesis.meanflow_contour(mat_out; clims=clim, cmap=cmap, levels=12, linewidth= 0.1,colorbar=true)

#     # Left panel keeps its y-axis; right panel drops the y-axis (shared) and
#     # carries the single shared colorbar. Negative margins pull them together.
#     plot!(img_in;  right_margin=-1Plots.mm,)
#     plot!(img_out; left_margin=-1Plots.mm, ylabel="", yformatter=_ -> "",
#           link=:y)

#     # Only the bottom row (last channel) shows the x-axis; the upper rows drop
#     # their x label and tick labels so the stacked rows share one x-axis.
#     if ch == 1 
#         plot!(img_in,  top_margin=-2Plots.mm)
#         plot!(img_out, top_margin=-2Plots.mm)
#     end
#     if ch < 2
#         plot!(img_in;  xlabel="", xformatter=_ -> "",top_margin=-1.5Plots.mm, bottom_margin=-1Plots.mm)
#         plot!(img_out; xlabel="", xformatter=_ -> "",top_margin=-1.5Plots.mm, bottom_margin=-1Plots.mm)
#     end

#     plt_comb = plot(img_in, img_out;
#         layout=grid(1, 2; widths=[0.4425, 0.5575]),
#         link=:y, legend=false, size=(700, 350), dpi=600, grid=false)
#     if ch == 1
#         plot!(plt_comb, bottom_margin=-16Plots.mm)
#     end
#     # display(plt_comb)

#     push!(plots, plt_comb)

#     # Signed reconstruction error, symmetric diverging map so 0 = white.
#     mat_diff = abs.(mat_in .- mat_out)
#     mat_diff ./= maximum(mat_diff)
#     # dclim = Thesis.sym_clims(mat_diff)
#     dclim = (0, 1)
#     img_diff = Thesis.meanflow_contour(mat_diff; clims=dclim, cmap=cmap,
#                                        levels=48, linewidth=0.0, colorbar=true)
#     if ch == 1
#          plot!(img_diff; colorbar=false)
#     end
#     if ch == 2
#         plot!(img_diff; ylabel="", yformatter=_ -> "",
#               top_margin=-2Plots.mm, bottom_margin=-1Plots.mm)
#     end
#     push!(diff_plots, img_diff)
# end

# p = plot(plots...;
#              layout=(2, 1),
#              link=:none, legend=false,
#              size=(700, 700),
#              dpi=600, grid=false)
# display(p)

# savefig(p, joinpath(model1, "velocity_comp.png"))

# p_diff = plot(diff_plots...;
#         layout=grid(1, 2; widths=[0.4425, 0.5575]),
#         link=:y, legend=false, size=(700, 350), dpi=600, grid=false)

# display(p_diff)

# savefig(p_diff, joinpath(model1, "velocity_diff.png"))

# ##
# # display(p_rec)
# # savefig(p, joinpath(model1, "loss_evolution.pdf"))
# # load one model's loss trajectory, expressed in epochs
# function load_loss_segment(model_dir)
#     loss_path = joinpath(model_dir, "loss_trajectory.jld2")
#     ckpt_path = joinpath(model_dir, "checkpoint.jld2")
#     isfile(loss_path) || error("missing loss_trajectory.jld2 in $model_dir")
#     isfile(ckpt_path) || error("missing checkpoint.jld2 in $model_dir")

#     losses = JLD2.load(loss_path)
#     train_losses  = get(losses, "train_losses", Float32[])
#     rec_losses    = get(losses, "rec_losses", Float32[])
#     val_losses    = get(losses, "val_losses", Float32[])
#     test_losses   = get(losses, "test_losses", Float32[])
#     div_losses    = get(losses, "div_losses", Float32[])
#     curl_losses   = get(losses, "curl_losses", Float32[])
#     strain_losses = get(losses, "strain_losses", Float32[])
#     iters         = get(losses, "iters", Int[])
#     val_iters     = get(losses, "val_iters", Int[])

#     checkpoint = JLD2.load(ckpt_path)
#     args = LuxArgs(; checkpoint["args"]...)

#     if isempty(iters) && !isempty(train_losses)
#         iters = collect(1:length(train_losses))
#     end

#     iters_per_epoch = if !isempty(val_iters)
#         Float64(val_iters[1])
#     elseif !isempty(iters)
#         Float64(length(iters) / max(args.epochs, 1))
#     else
#         1.0
#     end

#     train_epochs = iters ./ iters_per_epoch
#     val_epochs   = val_iters ./ iters_per_epoch
#     return (; train_epochs, train_losses, rec_losses, val_epochs, val_losses, test_losses,
#               div_losses, curl_losses, strain_losses)
# end

# segs = load_loss_segment.(models)

# # stitch the per-model epoch axes end to end. each segment is offset by the
# # cumulative number of epochs trained so far, so the x axis is "combined epochs".
# seg_widths = [maximum(vcat(s.train_epochs, s.val_epochs)) for s in segs]
# offsets    = vcat(0.0, cumsum(seg_widths)[1:end-1])
# boundaries = cumsum(seg_widths)          # x positions where one model hands off to the next

# # y ticks: powers of 10 spanning the data
# positive_vals = Float64[]
# for s in segs, v in vcat(s.train_losses, s.rec_losses, s.val_losses, s.test_losses,
#                           s.div_losses, s.curl_losses, s.strain_losses)
#     (v > 0 && isfinite(v)) && push!(positive_vals, v)
# end
# ylo = isempty(positive_vals) ? -5 : floor(Int, log10(minimum(positive_vals)))
# yticks_pow10 = 10.0 .^ (ylo:0)

# # concatenate the three segments into single train/val/test series, inserting a
# # NaN between segments so the line breaks cleanly at each model boundary while
# # still being one legend entry / one colour.
# function stitch(getx, gety)
#     xs = Float64[]; ys = Float64[]
#     for (off, s) in zip(offsets, segs)
#         x = getx(s); y = gety(s)
#         isempty(y) && continue
#         append!(xs, x .+ off); append!(ys, y)
#         push!(xs, NaN); push!(ys, NaN)
#     end
#     return xs, ys
# end

# train_x, train_y = stitch(s -> s.train_epochs, s -> s.train_losses)
# rec_x,   rec_y   = stitch(s -> s.train_epochs, s -> s.rec_losses)
# val_x,   val_y   = stitch(s -> s.val_epochs,   s -> s.val_losses)
# test_x,  test_y  = stitch(s -> s.val_epochs,   s -> s.test_losses)

# # physics residuals are logged every train iteration, so they share train_epochs
# div_x,    div_y    = stitch(s -> s.train_epochs, s -> s.div_losses)
# curl_x,   curl_y   = stitch(s -> s.train_epochs, s -> s.curl_losses)
# strain_x, strain_y = stitch(s -> s.train_epochs, s -> s.strain_losses)

# hasdata(y) = any(v -> v > 0 && isfinite(v), y)

# p = plot(
#     yscale = :log10,
#     yticks = yticks_pow10,
#     minorgrid = true,
#     minor_ticks = true,
#     gridalpha = 0.2,
#     gridlinewidth = 0.4,
#     grid = :y,
#     framestyle = :box,
#     foreground_color_legend = :black,
#     background_color_legend = RGBA(1, 1, 1, 0.8),
#     dpi = 500,
#     size = (700, 300),
#     titlefontsize = 12,
#     guidefontsize = 10,
#     tickfontsize = 8,
#     legendfontsize = 7,
#     left_margin = 3Plots.mm,
#     right_margin = 8Plots.mm,   # match AE_loss_plot (room for twin CC axis)
#     top_margin = 1Plots.mm,
#     bottom_margin = 2Plots.mm,
#     ylims = (-Inf, 1),
#     legend = :bottomleft,
# )

# # dotted dividers between models (skip the final boundary = end of plot)
# for b in boundaries[1:end-1]
#     vline!(p, [b]; ls = :dot, lw = 1, color = :gray40, label = "")
# end

# # label each model's region near the top of the plot
# for (off, w, lab) in zip(offsets, seg_widths, model_labels)
#     annotate!(p, off + w / 2, 0.7, text(lab, 8, :gray30, :center))
# end

# plot!(p, train_x, train_y; label = "train", xlabel = "Combined epoch",
#       ylabel = L"$\mathcal{L}$", lw = 1.2, color = :black)



# # final val/test losses from the end of the last segment (2nd retrain)
# final_val  = isempty(segs[end].val_losses)  ? NaN : segs[end].val_losses[end]
# final_test = isempty(segs[end].test_losses) ? NaN : segs[end].test_losses[end]

# !isempty(val_y)  && plot!(p, val_x,  val_y;
#                           label = @sprintf("val (final %.3g)", final_val),
#                           lw = 1, color = :red)
# hasdata(rec_y) && plot!(p, rec_x, rec_y; label = "rec", lw = 1, color = :teal)
# !isempty(test_y) && plot!(p, test_x, test_y;
#                           label = @sprintf("test (final %.3g)", final_test),
#                           lw = 1, color = :blue)

# # physics residuals


# hasdata(div_y)    && plot!(p, div_x,    div_y;    label = L"\nabla\cdot u",
#                            lw = 0.8, alpha = 0.9, color = :purple)
# hasdata(curl_y)   && plot!(p, curl_x,   curl_y;   label = L"\omega",
#                            lw = 0.8, alpha = 0.9, color = :orange)
# hasdata(strain_y) && plot!(p, strain_x, strain_y; label = L"\mathrm{strain}",
#                            lw = 0.8, alpha = 0.9, ls = :dashdot, color = :green)

# # AE_loss_plot carries a twin CC axis (ylims 0–1) on the right, which narrows
# # its main plot box. Add a matching twin axis here so the plotting area has
# # identical dimensions and the two figures align in LaTeX — but paint it white
# # (ticks, tick labels, guide) so it reserves the space invisibly.
# plot!(p; legendcolumns=2, legendfontsize=5)
# p2 = twinx(p)
# plot!(p2; ylabel = "CC", ylims = (0, 1), guidefontsize = 10, tickfontsize = 8,
#       legend = false,
#       foreground_color_axis = :white,
#       foreground_color_border = :white,
#       foreground_color_text = :white,
#       foreground_color_guide = :white)

# display(p)

# out_path = joinpath(basedir, "combined_loss_evolution.pdf")
# savefig(p, out_path)
# @info "saved" out_path
# nothing