using WaterLily
using Plots
using JLD2

@load "data/latent_data/128_RHS_biot_data_arr_force_period.jld2" z

println(size(z))
typeof(z) == Vector{Vector{Float32}}  # or maybe Matrix{Float32} (64×179)

Z = hcat(z...)  # concatenates all 64-vectors column-wise
@show size(Z)   # (64,179)

plot(Z'; xlabel="Snapshot index", ylabel="Latent value",
     title="Latent dimensions over time",
     legend=false, lw=0.8)

using FFTW
latent_series = Z[1, :] .- mean(Z[1, :])
freq = abs.(fft(latent_series))
idx = argmax(freq)
println("index of maximum: ", idx, "  value: ", freq[idx])
plot(freq[1:div(end,2)]; title="Latent spectrum of dim1",
     xlabel="frequency index", ylabel="|FFT|")