using JLD2
using Statistics
using Plots
using Thesis
using LaTeXStrings
using FFTW
using DSP


basedir = "data/saved_models/inline_runs_hpc/base"
model1 = "$basedir/AE_Jun05-1553__E500_HW256x256_C4to2_nc5_nd1_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0"
model2 = "$basedir/AE_Jun05-1610__E100_HW256x256_C4to2_nc5_nd1_z16_C8_lr0p0002_wd0p0009_bs16_NY_LL1_Tl0p0"
model3 = "$basedir/AE_Jun05-1617__E100_HW256x256_C4to2_nc5_nd1_z16_C8_lr0p0002_wd0p0009_bs16_NY_LL1_Tl0p0"

models = [model1, model2, model3]
model_labels = ["initial", "retrain 1", "retrain 2"]

# load one model's loss trajectory, expressed in epochs
function load_loss_segment(model_dir)
    loss_path = joinpath(model_dir, "loss_trajectory.jld2")
    ckpt_path = joinpath(model_dir, "checkpoint.jld2")
    isfile(loss_path) || error("missing loss_trajectory.jld2 in $model_dir")
    isfile(ckpt_path) || error("missing checkpoint.jld2 in $model_dir")

    losses = JLD2.load(loss_path)
    train_losses  = get(losses, "train_losses", Float32[])
    val_losses    = get(losses, "val_losses", Float32[])
    test_losses   = get(losses, "test_losses", Float32[])
    div_losses    = get(losses, "div_losses", Float32[])
    curl_losses   = get(losses, "curl_losses", Float32[])
    strain_losses = get(losses, "strain_losses", Float32[])
    iters         = get(losses, "iters", Int[])
    val_iters     = get(losses, "val_iters", Int[])

    checkpoint = JLD2.load(ckpt_path)
    args = LuxArgs(; checkpoint["args"]...)

    if isempty(iters) && !isempty(train_losses)
        iters = collect(1:length(train_losses))
    end

    iters_per_epoch = if !isempty(val_iters)
        Float64(val_iters[1])
    elseif !isempty(iters)
        Float64(length(iters) / max(args.epochs, 1))
    else
        1.0
    end

    train_epochs = iters ./ iters_per_epoch
    val_epochs   = val_iters ./ iters_per_epoch
    return (; train_epochs, train_losses, val_epochs, val_losses, test_losses,
              div_losses, curl_losses, strain_losses)
end

segs = load_loss_segment.(models)

# stitch the per-model epoch axes end to end. each segment is offset by the
# cumulative number of epochs trained so far, so the x axis is "combined epochs".
seg_widths = [maximum(vcat(s.train_epochs, s.val_epochs)) for s in segs]
offsets    = vcat(0.0, cumsum(seg_widths)[1:end-1])
boundaries = cumsum(seg_widths)          # x positions where one model hands off to the next

# y ticks: powers of 10 spanning the data
positive_vals = Float64[]
for s in segs, v in vcat(s.train_losses, s.val_losses, s.test_losses,
                          s.div_losses, s.curl_losses, s.strain_losses)
    (v > 0 && isfinite(v)) && push!(positive_vals, v)
end
ylo = isempty(positive_vals) ? -5 : floor(Int, log10(minimum(positive_vals)))
yticks_pow10 = 10.0 .^ (ylo:0)

# concatenate the three segments into single train/val/test series, inserting a
# NaN between segments so the line breaks cleanly at each model boundary while
# still being one legend entry / one colour.
function stitch(getx, gety)
    xs = Float64[]; ys = Float64[]
    for (off, s) in zip(offsets, segs)
        x = getx(s); y = gety(s)
        isempty(y) && continue
        append!(xs, x .+ off); append!(ys, y)
        push!(xs, NaN); push!(ys, NaN)
    end
    return xs, ys
end

train_x, train_y = stitch(s -> s.train_epochs, s -> s.train_losses)
val_x,   val_y   = stitch(s -> s.val_epochs,   s -> s.val_losses)
test_x,  test_y  = stitch(s -> s.val_epochs,   s -> s.test_losses)

# physics residuals are logged every train iteration, so they share train_epochs
div_x,    div_y    = stitch(s -> s.train_epochs, s -> s.div_losses)
curl_x,   curl_y   = stitch(s -> s.train_epochs, s -> s.curl_losses)
strain_x, strain_y = stitch(s -> s.train_epochs, s -> s.strain_losses)

hasdata(y) = any(v -> v > 0 && isfinite(v), y)

p = plot(
    yscale = :log10,
    yticks = yticks_pow10,
    minorgrid = true,
    minor_ticks = true,
    gridalpha = 0.2,
    gridlinewidth = 0.4,
    grid = :y,
    framestyle = :box,
    foreground_color_legend = :black,
    background_color_legend = RGBA(1, 1, 1, 0.8),
    dpi = 500,
    size = (800, 320),
    titlefontsize = 12,
    guidefontsize = 10,
    tickfontsize = 8,
    legendfontsize = 7,
    left_margin = 3Plots.mm,
    right_margin = 4Plots.mm,
    top_margin = 4Plots.mm,
    bottom_margin = 3Plots.mm,
    ylims = (-Inf, 1),
    legend = :bottomleft,
)

# dotted dividers between models (skip the final boundary = end of plot)
for b in boundaries[1:end-1]
    vline!(p, [b]; ls = :dot, lw = 1, color = :gray40, label = "")
end

# label each model's region near the top of the plot
for (off, w, lab) in zip(offsets, seg_widths, model_labels)
    annotate!(p, off + w / 2, 0.7, text(lab, 8, :gray30, :center))
end

plot!(p, train_x, train_y; label = "train", xlabel = "Combined epoch",
      ylabel = L"$\mathcal{L}$", lw = 1.2, color = :black)

!isempty(val_y)  && plot!(p, val_x,  val_y;  label = "val",  lw = 1, color = :red)
!isempty(test_y) && plot!(p, test_x, test_y; label = "test", lw = 1, color = :blue)

# physics residuals
hasdata(div_y)    && plot!(p, div_x,    div_y;    label = L"|\nabla\cdot u|",
                           lw = 0.8, alpha = 0.9, color = :purple)
hasdata(curl_y)   && plot!(p, curl_x,   curl_y;   label = L"\omega\ \mathrm{loss}",
                           lw = 0.8, alpha = 0.9, color = :orange)
hasdata(strain_y) && plot!(p, strain_x, strain_y; label = L"\mathrm{strain}",
                           lw = 0.8, alpha = 0.9, ls = :dashdot, color = :green)

display(p)

out_path = joinpath(basedir, "combined_loss_evolution.png")
savefig(p, out_path)
@info "saved" out_path
