using JLD2

@load "matth/data/RHS_shedding_data_arr.jld2" RHS_data

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

vel_mag = sqrt.(sum(random_matrix .^ 2, dims=3))
# Make a contour plot
px = Plots.contourf(random_matrix[:,:,1]; title="x component of random RHS matrix", cmap=:viridis)
py = Plots.contourf(random_matrix[:,:,2]; title="y component of random RHS matrix", cmap=:viridis)
pmag = Plots.contourf(vel_mag[:,:,1]; title="Magnitude of random RHS matrix", cmap=:viridis)

plot(px, py, pmag, layout=(3, 1), size=(500, 750))