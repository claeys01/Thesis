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
includet("../simulations/vortex_shedding_biot_savart.jl")

function physical_ass(AE_path::String, NODE_path::String; saveplot=false)
    sim = circle_shedding_biot(;Re=2500, n=2^8, m=2^8, perturb=false)

    # load AE
    (_, dec, _, ps, st, ae_args)  = load_trained_AE(AE_path; return_params=true)

    normalizer = load_normalizer(AE_path)

    # # load NODE
    node, node_args = load_node(NODE_path)

    # # load data
    # #   load AE data
    # simdata = load_simdata(ae_args.full_data_path)
    # preprocess_data!(simdata)
    # N = size(simdata.u, 4)
    # @show N

    #   load node data
    _, t_train,  _, _       = get_NODE_data(node_args.train_latent_path;  downsample=-1,  verbose=false)    
    _, t_test,  _, z0_test  = get_NODE_data(node_args.test_latent_path;  downsample=-1,  verbose=false)    
    _, t_total, _, z0_total = get_NODE_data(node_args.total_latent_path; downsample=-1, verbose=false)

    # predict latents
    ẑ_total, ẑ_test = predict_array(node, z0_total; t=t_total), predict_array(node, z0_test; t=t_test)
    @show size(ẑ_total), size(ẑ_test)
    
    # downsample data
    n_down = 200
    total_downsample = downsample_equal(collect(1:size(ẑ_total, 2)), n_down)
    test_downsample = downsample_equal(collect(1:size(ẑ_test, 2)), n_down)

    ẑ_total, ẑ_test = ẑ_total[:, total_downsample], ẑ_test[:, test_downsample]
    t_total, t_test = t_total[total_downsample], t_test[test_downsample]
    @show size(ẑ_total), size(ẑ_test)

    # load flow fields
    simdata = load_simdata(ae_args.full_data_path)
    preprocess_data!(simdata)
    # u = simdata.u[:, :, :, total_downsample]
    u = impose_biot_bc_on_snapshot(simdata.u[:, :, :, total_downsample])
    # simdata = nothing

    u_div = div_field(u; avg=true)
    u_ε = kinetic_energy_dissipation(u; ν=sim.flow.ν, avg=true)

    # predict flow fields and impose BCs
    (û_total, _), (û_test, _) = dec(ẑ_total, ps.decoder, st.decoder), dec(ẑ_test, ps.decoder, st.decoder)

    û_total, û_test = denormalize_batch(û_total, normalizer).* simdata.μ₀[:, :, :, total_downsample] , denormalize_batch(û_test, normalizer) .* simdata.μ₀[:, :, :, total_downsample]
    
    plt1 = flood(û_test[:, :, 1, 100]; clims=(-0.3,1.75))
    û_total, û_test = impose_biot_bc_on_snapshot(û_total), impose_biot_bc_on_snapshot(û_test)
    plt2 = flood(û_test[:, :, 1, 100], clims=(-0.3,1.75))
    display(plot(plt1, plt2))

    # û_total, û_test = denormalize_batch(û_total, normalizer), denormalize_batch(û_test, normalizer)
    # û_total, û_test = impose_biot_bc_on_snapshot(û_total).* simdata.μ₀[:, :, :, total_downsample], impose_biot_bc_on_snapshot(û_test).* simdata.μ₀[:, :, :, total_downsample]
    
    û_div_total, û_div_test = div_field(û_total; avg=true), div_field(û_test; avg=true)
    û_ε_total, û_ε_test = kinetic_energy_dissipation(û_total; ν=sim.flow.ν, avg=true), kinetic_energy_dissipation(û_test; ν=sim.flow.ν, avg=true)

    div_plot = plot(title="Mean divergence", xlabel="time", ylabel="‖∇̇·u‖", grid=true, minorgrid=true)
    plot!(div_plot, t_total, u_div;       label="‖∇̇·u‖ (ground truth)", color=:black)
    plot!(div_plot, t_total, û_div_total; label="‖∇·û‖ (total)",        color=:blue)
    plot!(div_plot, t_test, û_div_test;   label="‖∇·û‖ (test)",         color=:red)
    region_spans!(div_plot, t_train, t_test)

    ε_plot = plot(title="TKE dissipation", xlabel="time", ylabel="‖ε‖", grid=true, minorgrid=true)
    plot!(ε_plot, t_total, u_ε;       label="‖ε‖ (ground truth)", color=:black)
    plot!(ε_plot, t_total, û_ε_total; label="‖ε‖ (total)",        color=:blue)
    plot!(ε_plot, t_test,  û_ε_test;  label="‖ε‖ (test)",         color=:red)
    region_spans!(ε_plot, t_train, t_test)

    return plot(div_plot, ε_plot; layout=(2, 1), size=(900, 900), )

end
if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    node_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"
    AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"
    physical_ass(AE_path, node_path)
end