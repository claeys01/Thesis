using Thesis
using Statistics
using Plots
using CUDA


root_path = is_hpc() ? "/scratch/mfbclaeys" : ""
AE_path_tl1 = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
AE_path_tl1 = joinpath(root_path, AE_path_tl1)

normalizer = load_normalizer(AE_path_tl1)
ae_bundle, ae_args = load_trained_AE(AE_path_tl1)

device = gpu_device()
batch_sizes = [1, 5, 10, 20, 30, 50]
H, W, C = ae_args.input_dim

println("Benchmarking AE forward pass (encode + decode)")
println("Input size: ($H, $W, $C, batch)")
println("-"^50)

ae_times = Float64[]
for bs in batch_sizes
    x = device(randn(Float32, H, W, C, bs))

    # warmup
    ae_bundle.ae(x, ae_bundle.ps, ae_bundle.st)

    n_runs = 10
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed ae_bundle.ae(x, ae_bundle.ps, ae_bundle.st)
        push!(times, t)
    end
    mean_t = mean(times) * 1000
    std_t = std(times) * 1000
    push!(ae_times, mean_t)
    println("batch=$bs  =>  $(round(mean_t; digits=2)) ± $(round(std_t; digits=2)) ms  (per sample: $(round(mean_t / bs; digits=2)) ms)")
end

println("\n\nBenchmarking encoder only")
println("Input size: ($H, $W, $C, batch)")
println("-"^50)

enc_times = Float64[]
for bs in batch_sizes
    x = device(randn(Float32, H, W, C, bs))

    # warmup
    ae_bundle.ae.encoder(x, ae_bundle.ps.encoder, ae_bundle.st.encoder)

    n_runs = 10
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed ae_bundle.ae.encoder(x, ae_bundle.ps.encoder, ae_bundle.st.encoder)
        push!(times, t)
    end
    mean_t = mean(times) * 1000
    std_t = std(times) * 1000
    push!(enc_times, mean_t)
    println("batch=$bs  =>  $(round(mean_t; digits=2)) ± $(round(std_t; digits=2)) ms  (per sample: $(round(mean_t / bs; digits=2)) ms)")
end

println("\n\nBenchmarking decoder only")
println("Latent size: ($(ae_args.latent_dim), batch)")
println("-"^50)

dec_times = Float64[]
for bs in batch_sizes
    z = device(randn(Float32, ae_args.latent_dim, bs))

    # warmup
    ae_bundle.ae.decoder(z, ae_bundle.ps.decoder, ae_bundle.st.decoder)

    n_runs = 10
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed ae_bundle.ae.decoder(z, ae_bundle.ps.decoder, ae_bundle.st.decoder)
        push!(times, t)
    end
    mean_t = mean(times) * 1000
    std_t = std(times) * 1000
    push!(dec_times, mean_t)
    println("batch=$bs  =>  $(round(mean_t; digits=2)) ± $(round(std_t; digits=2)) ms  (per sample: $(round(mean_t / bs; digits=2)) ms)")
end

plt = plot(batch_sizes, ae_times, label="Full AE", marker=:circle, lw=2)
plot!(plt, batch_sizes, enc_times, label="Encoder", marker=:square, lw=2)
plot!(plt, batch_sizes, dec_times, label="Decoder", marker=:diamond, lw=2)
xlabel!(plt, "Batch size")
ylabel!(plt, "Mean time [ms]")
title!(plt, "AE inference time vs batch size")
display(plt)
savefig(plt, "figs/inference_time.png")

println(dec_times ./ enc_times)