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

function RHS(flow::Flow{N};λ=WaterLily.quick,udf=nothing,kwargs...) where N
    RHS = WaterLily.conv_diff!(flow.f,flow.u⁰,flow.σ,λ;ν=flow.ν,perdir=flow.perdir) - grad(flow.p)
    return RHS
end

function downsample_RHS_data!(RHS_data; tmin=50, tmax=75, n_samples=50)
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

    println("Downsampled to ", length(RHS_data["time"]), " time steps.")
    println("Input data size: ", size(RHS_data["RHS"][1]))
end