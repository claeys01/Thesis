using Thesis
using Statistics
using Plots
using CUDA
using Lux
using Random
using JLD2
using Printf

# tune_dict = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/tune_models"
# out_csv = joinpath(tune_dict, "tune_summary_raw.csv")

args = LuxArgs()
enc = Encoder(args)
dec = Decoder(args)
ae = AE(enc, dec)



# function build_test_set(args)
#     data, loaders, _ = Thesis.get_data(args.batch_size, args.full_data_path;
#         n_training=args.train_downsample, n_test=args.test_downsample,
#         split=args.split, t_training=args.t_training,
#         verbose=false)
#     TestData = data.TestData
#     test_loader = loaders.test_loader
#     data = nothing; loaders = nothing
#     clear_memory!()
#     return TestData, test_loader
# end

# function evaluate_checkpoint(ckpt_path::String, TestData, test_loader)
#     dev = Thesis.get_device()
#     ae_bundle, args = load_trained_AE(ckpt_path; device=dev, return_params=true)
#     ae, ps, st = ae_bundle.ae, ae_bundle.ps, ae_bundle.st
#     normalizer = load_normalizer(ckpt_path)

#     rec_sum = 0.0; div_sum = 0.0; curl_sum = 0.0; tke_sum = 0.0; n = 0
#     for idx in test_loader
#         x_in, x_target, μ₀ = Thesis.build_batch(TestData, idx)
#         x_in_f = Float32.(x_in)
#         if args.normalize
#             uvc = x_in_f[:, :, 1:2, :]
#             uvc_norm, _ = normalize_batch(uvc; normalizer=normalizer)
#             x_in_f = cat(uvc_norm, x_in_f[:, :, 3:4, :]; dims=3)
#         end
#         x_in_dev = dev(x_in_f)
#         x̂_norm, _ = ae(x_in_dev, ps, st)
#         x̂ = Array(denormalize_batch(cpu_device()(x̂_norm), normalizer)) .* Array(μ₀)
#         x = Array(x_target)

#         rec_sum  += mean(abs, x .- x̂)
#         div_sum  += mean(abs, div_field(x̂; buff=1))
#         curl_sum += mean(abs, curl_vectorized(x; buff=1) .- curl_vectorized(x̂; buff=1))
#         tke_x = redirect_stdout(devnull) do
#             kinetic_energy_dissipation(x; ν=1.0, avg=true, buff=1)
#         end
#         tke_x̂ = redirect_stdout(devnull) do
#             kinetic_energy_dissipation(x̂; ν=1.0, avg=true, buff=1)
#         end
#         tke_sum += mean(abs, tke_x .- tke_x̂)
#         n += 1
#     end
#     n = max(n, 1)
#     return (recon_mae=rec_sum/n, divergence=div_sum/n,
#             curl_err=curl_sum/n, tke_err=tke_sum/n)
# end

# function tag_for(args)
#     return @sprintf("d%g_c%g_z%d_nc%d_nd%d",
#         args.λdiv, args.λcurl, args.latent_dim, args.n_conv, args.n_dense)
# end

# function write_rows(rows::Vector{<:NamedTuple}, csv_path::String)
#     cols = [:tag, :λdiv, :λcurl, :latent_dim, :n_conv, :n_dense,
#             :recon_mae, :divergence, :curl_err, :tke_err, :path]
#     open(csv_path, "w") do io
#         println(io, join(string.(cols), ","))
#         for r in rows
#             println(io, join((string(getfield(r, c)) for c in cols), ","))
#         end
#     end
#     @info "CSV written to $csv_path ($(length(rows)) rows)"
# end

# function main()
#     subdirs = filter(isdir, [joinpath(tune_dict, d) for d in readdir(tune_dict)])
#     ckpts = filter(isfile, [joinpath(d, "checkpoint.jld2") for d in subdirs])
#     @info "Found $(length(ckpts)) checkpoints in $tune_dict"
#     isempty(ckpts) && (@warn "no checkpoints found"; return)

#     # build test snapshots ONCE using the first checkpoint's data args
#     _, init_args = load_trained_AE(ckpts[1]; device=cpu_device(), return_params=true)
#     @info "Building TestData once" t_training=init_args.t_training test_downsample=init_args.test_downsample split=init_args.split
#     TestData, test_loader = build_test_set(init_args)

#     rows = NamedTuple[]
#     for (i, ckpt) in enumerate(ckpts)
#         @info "[$i/$(length(ckpts))] evaluating: $(basename(dirname(ckpt)))"
#         clear_memory!()
#         try
#             _, args = load_trained_AE(ckpt; device=cpu_device(), return_params=true)
#             tag = tag_for(args)
#             m = evaluate_checkpoint(ckpt, TestData, test_loader)
#             push!(rows, (tag=tag, λdiv=args.λdiv, λcurl=args.λcurl,
#                          latent_dim=args.latent_dim, n_conv=args.n_conv, n_dense=args.n_dense,
#                          recon_mae=m.recon_mae, divergence=m.divergence,
#                          curl_err=m.curl_err, tke_err=m.tke_err, path=ckpt))
#             @info "  ✓ recon=$(m.recon_mae) div=$(m.divergence) curl=$(m.curl_err) tke=$(m.tke_err)"
#         catch e
#             @error "Run failed for $ckpt" exception=(e, catch_backtrace())
#         end
#         !isempty(rows) && write_rows(rows, out_csv)
#     end

#     @info "Done. $(length(rows))/$(length(ckpts)) evaluated."
# end

# main()
