using WaterLily
import WaterLily: ∂, @loop

function grad(field::AbstractArray)
    T = eltype(field)
    sz = size(field)
    N = ndims(field)
    grad = zeros(T, sz..., N)  # e.g., zeros(Float32, Nx, Ny, 2)
    for n in 1:N
        @loop grad[Tuple(I)..., n] = ∂(n, I, field) over I ∈ inside(field)
    end
    return grad
end

function RHS(flow::Flow{N};λ=WaterLily.quick,kwargs...) where N
    RHS = WaterLily.conv_diff!(flow.f,flow.u⁰,flow.σ,λ;ν=flow.ν,perdir=flow.perdir) - grad(flow.p)
    return RHS
end

function downsample_RHS_data!(RHS_data; tmin=-1, tmax=-1, n_samples=-1, clip_bc=false)
    if n_samples == -1
        n_samples = length(RHS_data["time"])
    end
    if tmin == -1
        tmin = RHS_data["time"][1]
    end
    if tmax == -1
        tmax = RHS_data["time"][end]
    end 
    # Select indices corresponding to time tmin to tmax
    time_indices = findall(t -> t ≥ tmin && t ≤ tmax, RHS_data["time"])
    selected_indices = time_indices

    # Downsample to n_samples entries
    downsampled_indices = round.(Int, range(1, length(selected_indices), length=n_samples))
    final_indices = selected_indices[downsampled_indices]

    # Downsample all relevant entries in RHS_data
    RHS_data["time"] = RHS_data["time"][final_indices]
    RHS_data["Δt"] = RHS_data["Δt"][final_indices]
    RHS_data["RHS"] = RHS_data["RHS"][final_indices]

    if clip_bc
        # Exclude outer cells (boundary) from each RHS entry
        for i in 1:length(RHS_data["RHS"])
            arr = RHS_data["RHS"][i]
            # Assume arr is at least 2D; clip first and last index in each dimension
            clipped_arr = arr[2:end-1, 2:end-1, :]
            RHS_data["RHS"][i] = clipped_arr
        end
    end

    println("Downsampled to ", length(RHS_data["time"]), " time steps.")
    println("Input data size: ", size(RHS_data["RHS"][1]))
end