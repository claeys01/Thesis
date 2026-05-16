using Thesis
using WaterLily
using WaterLily: flood, body_plot!
using Plots
using JLD2
using Printf

n = 2^9
t_end = 5.0
plot_interval = 0.1

run_tag = "biot_n$(n)_t$(Int(t_end))"
figs_dir = joinpath("figs", run_tag)
contour_dir = joinpath(figs_dir, "contours")
data_dir = joinpath("data", "initial_fields")
mkpath(contour_dir)
mkpath(data_dir)

sim = circle_shedding_biot(; n=n, m=n, perturb=false)
u₀ = load_u0("data/initial_fields/u0_biot_n512_t50.jld2")
sim.flow.u .= u₀

times = Float32[]
forces = Vector{Vector{Float32}}()

next_plot = plot_interval
anim = Animation()

while sim_time(sim) < t_end
    sim_step!(sim)
    sim_info(sim)

    push!(times, Float32(sim_time(sim)))
    push!(forces, Thesis.get_forces(sim))

    if sim_time(sim) ≥ next_plot
        @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])

        plt = flood(sim.flow.σ |> Array, shift=(-2, -1.5), clims=(-8, 8),
            axis=([], false), background=:white, cfill=:seismic,
            legend=false, border=:none, dpi=200, size=(800, 800))
        body_plot!(sim)

        t_str = @sprintf("%.1f", next_plot)
        savefig(plt, joinpath(contour_dir, "shedding_t$(t_str).png"))
        frame(anim, plt)

        next_plot += plot_interval
    end
end

gif(anim, joinpath(figs_dir, "$(run_tag).gif"), fps=35)

Fx = [f[1] for f in forces]
Fy = [f[2] for f in forces]
force_plt = plot(times, Fx; label="Fx (drag)", xlabel="tU/L", ylabel="F / (½ L U²)",
    lw=1.5, dpi=200, size=(900, 500))
plot!(force_plt, times, Fy; label="Fy (lift)", lw=1.5)
savefig(force_plt, joinpath(figs_dir, "forces.png"))
@save joinpath(figs_dir, "forces.jld2") times forces

u_final_path = joinpath(data_dir, "u0_$(run_tag).jld2")
save_u0(u_final_path, Array(sim.flow.u))
@info "Saved final velocity field to $(u_final_path)"
@info "Saved contours to $(contour_dir), gif + forces to $(figs_dir)"
