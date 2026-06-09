using Thesis
using JLD2
using Statistics
using Plots
using Printf

# sweep_dir = "data/saved_models/inline_runs_hpc/gs_sweep_old"
sweep_dir = "/scratch/mfbclaeys/data/inline_runs/inline_sweep_final_v2_actual_rest"
fmt = "pdf"                                   # format for the saved panel figures
csv_path = joinpath(sweep_dir, "meanflow_errors.csv")

# each inline run lives in its own subdirectory holding a hybrid_state.jld2
run_dirs = filter(isdir, readdir(sweep_dir; join=true))

header = ["run",
          "u_mae", "u_corr", "v_mae", "v_corr",
          "uu_mae", "uu_corr", "vv_mae", "vv_corr", "uv_mae", "uv_corr"]
rows = Vector{String}[]

for run_dir in run_dirs
    hs_path = joinpath(run_dir, "hybrid_state.jld2")
    if !isfile(hs_path)
        @warn "no hybrid_state.jld2, skipping" run_dir
        continue
    end

    name = basename(run_dir)
    @info "processing" name
    local sim_meanflow, ref_meanflow
    @load hs_path sim_meanflow ref_meanflow

    # combined panel figures, saved into the run directory
    pm = plot_meanflow_comparison(sim_meanflow, ref_meanflow; savedir=joinpath(run_dir, "meanflow_panels"))
    # savefig(pm, joinpath(run_dir, "meanflow_panels.$(fmt)"))
    pr = plot_rst_comparison(sim_meanflow, ref_meanflow; savedir=joinpath(run_dir, "rst_panels"), style=:filled)
    # savefig(pr, joinpath(run_dir, "rst_panels.$(fmt)"))

    fe = meanflow_errors(sim_meanflow, ref_meanflow)
    fmtnum(x) = @sprintf("%.6g", x)
    push!(rows, [name,
                 fmtnum(fe.u.l1),  fmtnum(fe.u.ρ),
                 fmtnum(fe.v.l1),  fmtnum(fe.v.ρ),
                 fmtnum(fe.uu.l1), fmtnum(fe.uu.ρ),
                 fmtnum(fe.vv.l1), fmtnum(fe.vv.ρ),
                 fmtnum(fe.uv.l1), fmtnum(fe.uv.ρ)])
end

open(csv_path, "w") do io
    println(io, join(header, ","))
    for r in rows
        println(io, join(r, ","))
    end
end

@info "done" runs=length(rows) csv=csv_path
