function plot_losses(loss_trajectory_path::AbstractString, checkpoint_path::AbstractString)

   # ------------------
    # Load data
    # ------------------
    losses = JLD2.load(loss_trajectory_path)

    train_losses   = get(losses, "train_losses", Float32[])
    rec_losses     = get(losses, "rec_losses", Float32[])
    div_losses     = get(losses, "div_losses", Float32[])
    curl_losses    = get(losses, "curl_losses", Float32[])
    strain_losses  = get(losses, "strain_losses", Float32[])
    iters          = get(losses, "iters", Int[])
    train_corrs    = get(losses, "train_corrs", Vector{Float32}[])

    val_losses     = get(losses, "val_losses", Float32[])
    val_iters      = get(losses, "val_iters", Int[])
    val_corrs      = get(losses, "val_corrs", Vector{Float32}[])

    test_losses    = get(losses, "test_losses", Float32[])
    test_corrs     = get(losses, "test_corrs", Vector{Float32}[])


    checkpoint = JLD2.load(checkpoint_path)
    args = LuxArgs(; checkpoint["args"]...)

    # fallback x axis if missing
    if isempty(iters) && !isempty(train_losses)
        iters = collect(1:length(train_losses))
    end

    # convert iterations -> epochs.
    # val_iters[k] = (iter count at end of epoch k), so val_iters[1] = iters_per_epoch.
    iters_per_epoch = if !isempty(val_iters)
        Float64(val_iters[1])
    elseif !isempty(iters)
        Float64(length(iters) / max(args.epochs, 1))
    else
        1.0
    end
    train_epochs = iters ./ iters_per_epoch
    val_epochs   = val_iters ./ iters_per_epoch

    # explicit log-scale ticks at every power of 10 covering the data range
    positive_vals = Float64[]
    for s in (train_losses, val_losses, test_losses, div_losses, curl_losses, strain_losses)
        append!(positive_vals, (v for v in s if v > 0 && isfinite(v)))
    end
    ylo = isempty(positive_vals) ? -5 : floor(Int, log10(minimum(positive_vals)))
    yhi = 0   # top of plot is capped at 1 = 10^0
    yticks_pow10 = 10.0 .^ (ylo:yhi)

    # figure annotation text (final val loss)
    final_loss_str = if !isempty(val_losses)
        @sprintf("final val loss = %.3g", val_losses[end])
    else
        ""
    end

    final_test_loss_str = if !isempty(test_losses)
        @sprintf("final test loss = %.3g", test_losses[end])
    else
        ""
    end
    p = plot(
        yscale = :log10,
        yticks = yticks_pow10,
        minorgrid = true,
        minor_ticks = true,
        gridalpha  = 0.2,
        gridlinewidth = 0.4,
        grid = :y,
        foreground_color_legend = :black,
        background_color_legend = RGBA(1,1,1,0.8), # light transparent white
        framestyle = :box,
        dpi = 500,
        size = (700, 300),
        titlefontsize = 12,
        guidefontsize = 10,
        tickfontsize  = 8,
        legendfontsize = 6,
        foreground_color_axis = :black,
        foreground_color_text = :black,
        left_margin   = 3Plots.mm,
        right_margin  = 8Plots.mm,   # leave room for twin CC axis label
        top_margin    = 1Plots.mm,
        bottom_margin = 2Plots.mm,
        ylims = (-Inf, 1),  # cap the upper limit at 1
        )

    plot!(p,
        train_epochs, train_losses;
        label = "train",
        xlabel = "Epoch",
        ylabel = L"$\mathcal{L}$",
        lw = 1.2,
        color = :black,
    )

    # validation loss
    if !isempty(val_losses)
        plot!(p, val_epochs, val_losses;
            label = @sprintf("val (final %.3g)", val_losses[end]),
            lw = 1,
            color = :red,
            # alpha = 0.9,
        )
    end

    # test losses
    if !isempty(test_losses)
        plot!(p, val_epochs, test_losses;
                # linestyle=:dashdot,
                label = @sprintf("test (final %.3g)", test_losses[end]),
                lw = 1,
                color = :blue,
                # alpha = 0.9,
            )
    end

    # optional extra loss terms if enabled
    if args.λdiv != 0 && !isempty(div_losses)
        plot!(p, train_epochs, div_losses;
            label = L"\nabla \cdot u",
            lw = 0.8,
            # ls = :dot,
            color = :purple,
            alpha = 0.9,
        )
    end

    if args.λcurl != 0 && !isempty(curl_losses)
        plot!(p, train_epochs, curl_losses;
            label = L"\omega\ \mathrm{loss}",
            lw = 0.8,
            # ls = :dot,
            color = :orange,
            alpha = 0.9,
        )
    end

    if args.λstrain != 0 && !isempty(strain_losses)
        plot!(p, train_epochs, strain_losses;
            label = L"\mathrm{strain\ term}",
            lw = 0.8,
            ls = :dashdot,
            color = :blue,
            alpha = 0.9,
        )
    end

    # log scale + nice ticks
    # plot!(p;
    #     yscale = :log10,
    #     minorgrid = true,
    #     minor_ticks = true,
    #     grid = :y,
    #     framestyle = :box,
    # )
    # if final_loss_str != ""
    #     if final_test_loss_str != ""
    #         title = final_loss_str * ",  " * final_test_loss_str
    #         # plot!(p, title = title, titlefont = font(12))  # change 10 to desired point siz
    #     else
    #         title = final_loss_str
    #     end
    #     plot!(p, title = title, titlefont = font(12))  # change 10 to desired point siz
    # end


    # ------------------
    # Right axis: CC
    # ------------------
    # assume train_corrs is a vector of 2-element vectors (or tuples)
    # first.(train_corrs) and last.(train_corrs) was in your code, keep that
    if !isempty(train_corrs)
        cc1 = first.(val_corrs)
        cc2 = last.(val_corrs)

        p2 = twinx()
        plot!(p2, val_epochs, cc1;
            label = L"CC_u",
            lw = 1,
            color = "#0F7173",   # dark teal (colorblind-safe blue-yellow axis)
            ylabel = "CC",
            ylims = (0,1),
            guidefontsize = 10,
            tickfontsize  = 8,
            legendfontsize = 6,
            foreground_color_axis = :black,
            foreground_color_text = :black,
        )

        plot!(p2, val_epochs, cc2;
            label = L"CC_v",
            lw = 1,
            color = "#B8860B",   # dark goldenrod
            ylims = (0,1),
        )
    end

    if !isempty(test_corrs)
        test_cc1 = first.(test_corrs)
        test_cc2 = last.(test_corrs)

        # p2 = twinx()
        plot!(p2, val_epochs, test_cc1;
            label = L"\mathrm{test}\ CC_u",
            lw = 0.8,
            linestyle=:dashdot,
            color = "#0F7173",   # dark teal (colorblind-safe blue-yellow axis)
            ylims = (0,1),
        )

        plot!(p2, val_epochs, test_cc2;
            label = L"\mathrm{test}\ CC_v",
            lw = 0.8,
            linestyle=:dashdot,
            color = "#B8860B",   # dark goldenrod
            ylims = (0,1),
        )
    end
    # ------------------
    # Cosmetic annotations
    # ------------------
    # 1. remove big title from top, just annotate final loss small in plot area

    # shrink legend boxes (border thin)
    plot!(p; legend=:bottomleft, foreground_color_legend=:black,
             background_color_legend=RGBA(1,1,1,0.7))
    plot!(p2; legend=:bottomright,  foreground_color_legend=:black,
             background_color_legend=RGBA(1,1,1,0.7))

    
    return p

end


# checkpoint = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
# losses = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/loss_trajectory.jld2"

# p = plot_losses(losses, checkpoint)
# display(p)
# savefig(
# p, "data/Lux_models/2025-12-01_17-16-32/loss_evolution.png")