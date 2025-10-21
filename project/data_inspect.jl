using JLD2
using WaterLily
using Statistics

@load "data/RHS_biot_data_arr.jld2" RHS_data

# Select indices corresponding to time 50 to 75
time_indices = findall(t -> t ≥ 50 && t ≤ 75, RHS_data["time"])
selected_indices = time_indices

# Downsample to ~50 entries
n_samples = 50
downsampled_indices = round.(Int, range(1, length(selected_indices), length=n_samples))
final_indices = selected_indices[downsampled_indices]

# Downsample all relevant entries in RHS_data
RHS_data["time"] = RHS_data["time"][final_indices]
RHS_data["Δt"] = RHS_data["Δt"][final_indices]
RHS_data["RHS"] = RHS_data["RHS"][final_indices]

println("Downsampled to ", length(RHS_data["time"]), " time steps.")

# RHS_data["flattened"] = [vec(r) for r in RHS_data["RHS"]]

using Plots

# Select a random matrix from RHS_data["RHS"]
random_idx = rand(1:length(RHS_data["RHS"]))
random_matrix = RHS_data["RHS"][random_idx]
println("Randomly selected matrix at index $random_idx with size: ", size(random_matrix))

println("Matrix type: ", typeof(random_matrix))
println("Matrix element type: ", eltype(random_matrix))
println("matrix dimensions: ", size(random_matrix))
println("Mean of random matrix: ", mean(random_matrix))

function RHS_stats(RHS)
    return mean(RHS), std(RHS)
end

rand_u, (u_mean, u_std) = random_matrix[:,:,1], RHS_stats(random_matrix[:,:,1])
rand_v, (v_mean, v_std) = random_matrix[:,:,2], RHS_stats(random_matrix[:,:,2])
mag  = sqrt.(sum(random_matrix .^ 2, dims=3))
(mag_mean, mag_std) = RHS_stats(mag) 

px = flood(rand_u, border=:none, clims=(u_mean-u_std, u_mean+u_std))
py = flood(rand_v, border=:none, clims=(v_mean-v_std, v_mean+v_std))
pmag = flood(mag[:,:,1], border=:none, clims=(mag_mean-mag_std, mag_mean+mag_std))

plot(px, py, pmag, layout=(3, 1), size=(500, 750))