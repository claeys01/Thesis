using Thesis
using WaterLily
using Plots
using Printf

function make_sim_gif(; Re=2500, n=2^9, m=2^9, t_end=30.0, Δt=0.25,
                       sample_every=1, fps=20,
                       gif_path="gifs/shedding_Re$(Re)_$(n)x$(m).gif")

    sim = circle_shedding_biot(; Re=Re, n=n, m=m, mem=Array, Δt=Δt)
    @show sim.L
    mkpath(dirname(gif_path))
    anim = Plots.Animation()

    frame_count = 0
    step_count = 0
    while sim_time(sim) < t_end
        sim_step!(sim)
        sim_info(sim)
        step_count += 1
        step_count % sample_every == 0 || continue
        ω = curl_vectorized(sim.flow.u) *sim.L/sim.U

        # @show minimum(ω), maximum(ω)
        clim = maximum(ω)
        plt = flood(ω;
            clims=(-clim, clim),
            aspectratio=:equal,
            framestyle=:none,
            border=:none,
            colorbar=false,
            title=@sprintf("t = %.2f  |  Re = %d", sim_time(sim), Re),
            titlefontsize=10,
            dpi=150,
            size=(800, 800))

        frame(anim, plt)
        frame_count += 1
    end

    gif(anim, gif_path; fps=fps, show_msg=false)
    @info "Saved $frame_count frames to $gif_path"
    return sim, gif_path
end

if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    make_sim_gif(; Re=2500, n=2^9, m=2^9, sample_every=20, fps=30, t_end=30)
end
