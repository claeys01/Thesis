using Thesis
using Plots
using Statistics

function physical_ass(AE_path::String, NODE_path::String; saveplot=false)
    sim = circle_shedding_biot(;Re=2500, n=2^8, m=2^8, perturb=false)
    # @show size(sim.flow.p)
    # ins = inside(sim.flow.u)
    # @show typeof(ins)

    aenode = AENODE(AE_path, node_path)

    _, t_train, _, _ = Thesis.get_NODE_data(aenode.node_args.train_latent_path; downsample=-1,  verbose=false)    
    _, t_test,  _, _ = Thesis.get_NODE_data(aenode.node_args.test_latent_path;  downsample=-1,  verbose=false)   

    simdata = load_simdata(aenode.ae_args.full_data_path)

    u = simdata.u[:, :, :, 1]
    μ₀ = simdata.μ₀[:, :, :, 1]
    time = simdata.time
    t = time[1]

    n_pred = 300
    t_arr_pred = [5.33333*i/32 for i in 1:n_pred]

    û = predict_n(aenode, u, μ₀, n_pred, t;Δt=5.3f0, return_traj=true, impose_biot=false)
    @info "caclulating prediction statistics"
    û_div = dropdims(mean(Thesis.div_vectorized(û; buff=1); dims=(1,2)); dims=(1,2))
    û_ε = Thesis.kinetic_energy_dissipation(û; ν=sim.flow.ν, buff=1, avg=true)


    # downsample data
    total_downsample = Thesis.downsample_equal(collect(1:size(time, 1)), n_pred)
    t_total = time[total_downsample]
    @info "calculating database statistics"
    u_div = dropdims(mean(Thesis.div_vectorized(simdata.u[:, :, :, total_downsample]; buff=2); dims=(1,2)); dims=(1,2))
    u_ε = Thesis.kinetic_energy_dissipation(simdata.u[:, :, :, total_downsample]; ν=sim.flow.ν, buff=2, avg=true)

    div_plot = plot(title="Mean divergence", xlabel="time", ylabel="‖∇̇·u‖", grid=true, minorgrid=true)
    plot!(div_plot, t_total, u_div; label="‖∇̇·u‖ (ground truth)", color=:black)
    plot!(div_plot, t_arr_pred, û_div; label="‖∇·û‖ (total)",     color=:blue)
    Thesis.region_spans!(div_plot, t_train, t_test)

    ε_plot = plot(title="TKE dissipation", xlabel="time", ylabel="‖ε‖", grid=true, minorgrid=true)
    plot!(ε_plot, t_total, u_ε; label="‖ε‖ (ground truth)", color=:black)
    plot!(ε_plot, t_arr_pred, û_ε; label="‖ε‖ (total)",        color=:blue)
    Thesis.region_spans!(ε_plot, t_train, t_test)

    return plot(div_plot, ε_plot; layout=(2, 1), size=(900, 900), )   
end
if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    node_path = "data/saved_models/NODE/16/RE2500/E1000_MS_Adam_250/node_params.jld2"
    AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1/checkpoint.jld2"
    physical_ass(AE_path, node_path)
end