using JLD2
using Plots
using Random

# Point this at the run directory that contains U_hybrid_inline.jld2 / U_ref_inline.jld2
savedir = length(ARGS) >= 1 ? ARGS[1] : "data/inline_runs/2026-05-30_15-19"

hybrid_path = joinpath(savedir, "U_hybrid_inline.jld2")
ref_path    = joinpath(savedir, "U_ref_inline.jld2")

# Each file stores, per saved step i: u/$i (the raw flow.u, an (Nx,Ny,2) array with
# ghost cells, where the last axis is the velocity component), t/$i (sim time), and n
# (total number of records).
function load_field_series(path)
    jldopen(path, "r") do f
        n = f["n"]
        t = [Float32(f["t/$i"]) for i in 1:n]
        u = [f["u/$i"] for i in 1:n]   # Vector of (Nx,Ny,2) arrays
        return t, u
    end
end

t_hyb, u_hyb = load_field_series(hybrid_path)
t_ref, u_ref = load_field_series(ref_path)

@info "Loaded field series" n_hybrid=length(t_hyb) n_ref=length(t_ref) field_size=size(u_hyb[1])

# Pick one random spatial coordinate + velocity component, shared by both runs.
Nx, Ny, ncomp = size(u_hyb[1])
rng = MersenneTwister(42)
ix = rand(rng, 1:Nx)
iy = rand(rng, 1:Ny)
ic = rand(rng, 1:ncomp)
@info "Sampling velocity at" ix iy component=ic

# Extract the time series of that single coordinate for each simulation.
vel_hyb = [u_hyb[i][ix, iy, ic] for i in eachindex(u_hyb)]
vel_ref = [u_ref[i][ix, iy, ic] for i in eachindex(u_ref)]

plt = plot(t_ref, vel_ref;
    label="reference", color=:black, lw=1.5,
    xlabel="t", ylabel="u[$ix, $iy, comp $ic]",
    title="Velocity at random coordinate ($ix, $iy), component $ic",
    dpi=400, size=(900, 400), framestyle=:box)
plot!(plt, t_hyb, vel_hyb; label="hybrid", color=:red, lw=1.5, linestyle=:dash)

out_path = joinpath(savedir, "velocity_point_compare.png")
savefig(plt, out_path)
@info "Saved comparison plot to $out_path"
display(plt)
