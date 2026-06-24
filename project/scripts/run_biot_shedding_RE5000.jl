using Thesis
using WaterLily
using WaterLily: MeanFlow
using Plots
using JLD2
using Printf

n = 2^9
t_end = 50.0
plot_interval = 0.05   # smaller -> more frames -> smoother gif
fps = 20
Re = 5000

run_tag = "biot_n$(n)_Re$(Re)_t$(Int(t_end))"
figs_dir = joinpath("figs", run_tag)
mkpath(figs_dir)

sim = circle_shedding_biot(; Re=Re, n=n, m=n, perturb=true)
u₀ = load_u0("data/initial_fields/RE5000/2e9/u_0.jld2")
sim.flow.u .= u₀

next_plot = plot_interval
anim = Animation()   # frames are buffered to a temp dir on disk, not RAM
meanflow = MeanFlow(sim.flow; uu_stats=true)

while sim_time(sim) < t_end
    sim_step!(sim)
    sim_info(sim)

    if sim_time(sim) ≥ next_plot
        plt = Thesis.curl_contour(sim.flow.u; colorbar=false, L=sim.L)
        frame(anim, plt)
        WaterLily.update!(meanflow, sim.flow)
        next_plot += plot_interval
    end
end

gif(anim, joinpath(figs_dir, "$(run_tag).gif"); fps=fps)

plt = Thesis.curl_contour(sim.flow.u; colorbar=false, L=sim.L)

mf_u = meanflow.U[:, :, 1]
mf_v = meanflow.U[:, :, 2]
div = cgrad(:RdBu, rev=true)
plt_u = Thesis.meanflow_contour(mf_u; clims=Thesis.sym_clims(mf_u), cmap=div, title="⟨u⟩")
plt_v = Thesis.meanflow_contour(mf_v; clims=Thesis.sym_clims(mf_v), cmap=div, title="⟨v⟩")
plt_meanflow = plot(plt_u, plt_v; layout=(1, 2), size=(1000, 350), dpi=400)
savefig(plt_meanflow, joinpath(figs_dir, "meanflow_components.png"))

# @save "data/initial_fields/RE5000/2e9/u_0.jld2" sim.flow.u
save_u0("data/initial_fields/RE5000/2e9/u_0.jld2", Array(sim.flow.u))