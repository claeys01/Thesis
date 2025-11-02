using JLD2, Plots
using Printf

includet("../AE/AE_core.jl")

default(
    # fontfamily = "Computer Modern",         # looks LaTeX-y if you have it
    linewidth  = 1.0,
    guidefont  = font(10),
    tickfont   = font(8),
    legendfont = font(8),
    gridalpha  = 0.2,
    gridlinewidth = 0.4,
    foreground_color_legend = :black,
    background_color_legend = RGBA(1,1,1,0.8), # light transparent white
    size=(600,400),                         # nice aspect for papers
)

function plot_losses(loss_trajectory_path::AbstractString, checkpoint_path::AbstractString)

   # ------------------
    # Load data
    # ------------------
    losses = JLD2.load(loss_trajectory_path)

    train_losses   = get(losses, "train_losses", Float32[])
    rec_losses     = get(losses, "rec_losses", Float32[])
    div_losses     = get(losses, "div_losses", Float32[])
    inside_losses  = get(losses, "inside_losses", Float32[])
    iters          = get(losses, "iters", Int[])
    val_losses     = get(losses, "val_losses", Float32[])
    val_iters      = get(losses, "val_iters", Int[])
    train_corrs    = get(losses, "train_corrs", Vector{Float32}[])
    val_corrs      = get(losses, "val_corrs", Vector{Float32}[])

    checkpoint = JLD2.load(checkpoint_path)
    args = Args(; checkpoint["args"]...)

    # fallback x axis if missing
    if isempty(iters) && !isempty(train_losses)
        iters = collect(1:length(train_losses))
    end

    # figure annotation text (final val loss)
    final_loss_str = if !isempty(val_losses)
        @sprintf("final val loss = %.3g", val_losses[end])
    else
        ""
    end


    p = plot(
        iters, train_losses;
        label = "train",
        xlabel = "Iteration",
        ylabel = "Loss",
        lw = 1,
        color = :gray,
        # alpha = 0.8,
        xguidefont = font(10),
        yguidefont = font(10),
        xtickfont  = font(8),
        ytickfont  = font(8),
    )

    # validation loss
    if !isempty(val_losses)
        plot!(p, val_iters, val_losses;
            label = "val",
            lw = 1,
            color = :red,
            # alpha = 0.9,
        )
    end

    # optional extra loss terms if enabled
    if args.λdiv != 0 && !isempty(div_losses)
        plot!(p, iters, div_losses;
            label = "divergence term",
            lw = 0.8,
            ls = :dot,
            color = :purple,
            alpha = 0.9,
        )
    end

    if args.λmask != 0 && !isempty(inside_losses)
        plot!(p, iters, inside_losses;
            label = "inside-mask term",
            lw = 0.8,
            ls = :dashdot,
            color = :orange,
            alpha = 0.9,
        )
    end

# log scale + nice ticks
    plot!(p;
        yscale = :log10,
        minorgrid = true,
        minor_ticks = true,
        grid = :y,
        framestyle = :box,
    )
    if final_loss_str != ""
        # scatter!(p, Float32(maximum(iters)), Float32(last(train_losses)+0.01))
        annotate!(p, (maximum(iters), last(train_losses)+0.01,
                    text(final_loss_str, 8, :black, :right)))
    end


    # ------------------
    # Right axis: CC
    # ------------------
    # assume train_corrs is a vector of 2-element vectors (or tuples)
    # first.(train_corrs) and last.(train_corrs) was in your code, keep that
    if !isempty(train_corrs)
        cc1 = first.(train_corrs)
        cc2 = last.(train_corrs)

        p2 = twinx()
        plot!(p2, iters, cc1;
            label = "CCᵤ",
            lw = 0.8,
            color = :green,
            alpha = 0.9,
            ylabel = "CC",
            ylims = (0,1),
            yguidefont = font(10),
            ytickfont  = font(8),
        )

        plot!(p2, iters, cc2;
            label = "CCᵥ",
            lw = 0.8,
            color = :magenta,
            alpha = 0.9,
            ylims = (0,1),
        )
    end
    # ------------------
    # Cosmetic annotations
    # ------------------
    # 1. remove big title from top, just annotate final loss small in plot area



    # shrink legend boxes (border thin)
    plot!(p; legend = (0.9,0.3), legendfontsize=8, foreground_color_legend=:black,
             background_color_legend=RGBA(1,1,1,0.7))
    plot!(p2; legend = (0.9,0.9), legendfontsize=8, foreground_color_legend=:black,
             background_color_legend=RGBA(1,1,1,0.7))

    
    return p

end

# p = plot_losses("data/models/2025-11-02_13-08-15/loss_trajectory.jld2",
#                 "data/models/2025-11-02_13-08-15/checkpoint.jld2")
# display(p)
# savefig(p, "data/models/2025-11-02_13-08-15/loss_evolution.png")