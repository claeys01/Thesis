using JLD2
using Revise
using LinearAlgebra
using WaterLily, BiotSavartBCs
using Random
Random.seed!(42)


includet("../NODE/NODE_core.jl")
includet("../NODE/NODE_RE2500_extrapolate.jl")
includet("../AE/Lux_AE.jl")
includet("../custom.jl")
# includet("../simulations/vortex_shedding_biot_savart.jl")

function physical_ass(AE_path::String, NODE_path::String)
    # load AE
    (_, _, _, _, _, ae_args)  = load_trained_AE(AE_path; return_params=true)    
    
    simdata = load_simdata(ae_args.full_data_path)
    # preprocess_data!(simdata)
    u_0 = copy(simdata.u[:, :, :, 1])
    
    u_inside = u_0[2:end-1, 2:end-1, :]

    u_new = impose_biot_bc_on_snapshot(u_inside)

    s = strain_field(u_0)
    @show mean(s), std(s)

    s_new = strain_field(u_new)
    @show mean(s_new), std(s_new)

    s_diff = (s .- s_new)
    s_diff_proc = mean(s_diff)/mean(s)*100
    @show s_diff_proc
    u_0_x = u_0[:, :, 1]
    u_new_x = u_new[:, :, 1]
    u_diff = u_0 .- u_new
    @show mean(u_diff)
    clim_x = (-1,1)

    plot_0 = flood(u_0_x;  clims=clim_x, border=:none)
    plot_new = flood(u_new_x;  clims=clim_x, border=:none)

    plt = plot(plot_0, plot_new)
    # display(plt)

    # @show s
    ε_0 = kinetic_energy_dissipation(u_0)
    ε_new = kinetic_energy_dissipation(u_new)

    @show mean(ε_0), mean(ε_new)
    ε_diff_proc = (mean(ε_0) - mean(ε_new))/mean(ε_0)*100
    @show ε_diff_proc
    
    divergence_0 = div_field(u_0)
    divergence_new = div_field(u_new)
    @show mean(divergence_0), mean(divergence_new)
    nothing
end
if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    node_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"
    AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"
    physical_ass(AE_path, node_path)
end