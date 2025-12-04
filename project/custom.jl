using WaterLily
import WaterLily: ∂, @loop, @inside, inside_u
using JLD2
using Random
using Statistics
includet("utils/SimDataTypes.jl")

using .SimDataTypes


function scalar_grad(field::AbstractArray)
    T = eltype(field); sz = size(field); N = ndims(field)
    grad = zeros(T, sz..., N)  # e.g., zeros(Float32, Nx, Ny, 2)
    for n in 1:N
        @loop grad[Tuple(I)..., n] = ∂(n, I, field) over I ∈ inside(field)
    end
    return grad
end

function grad_p(flow::Flow)
    return scalar_grad(flow.p) .* flow.μ₀
end

function RHS(flow::Flow{N};λ=WaterLily.quick,kwargs...) where N
    RHS = WaterLily.conv_diff!(flow.f,flow.u⁰,flow.σ,λ;ν=flow.ν,perdir=flow.perdir) - grad_p(flow)
    return RHS
end

function remove_ghosts(snapshot::AbstractArray)
    return snapshot[2:end-1, 2:end-1, :, :]
end

function preprocess_data!(data::SimData;
                          verbose::Bool = true)

    time = data.time

    # --- boundary clipping on u and μ₀ ---
    @inline function clip_time_series(ts)
        H, W, C, T = size(ts)
        temp = similar(ts, eltype(ts), H-2, W-2, C, T)
        @inbounds for i in axes(ts, 4)
            temp[:, :, :, i] = remove_ghosts(ts[:, :, :, i])
        end
        return temp
    end

    data.u = clip_time_series(data.u)
    data.μ₀ = clip_time_series(data.μ₀)

    if verbose
        @info "removed ghost cells; input data size (u): $(size(data.u))"
    end

    return data
end


ispow2(n::Integer) = n > 0 && (n & (n - 1)) == 0

function strain_field(u)
    all_dims = size(u)                    
    if !ispow2(all_dims[1]) 
        u = dropdims(remove_ghosts(u), dims = 4)
    end
    Tp = eltype(u)
    D = ndims(u) - 1
    dims= size(u)
    spatial = Tuple(size(u)[1:end-1])
    Sfield = zeros(Tp, dims..., D) 
    @loop Sfield[I,:,:] .= S(I, u) over I ∈ CartesianIndices(spatial)

    return Sfield   
end

function kinetic_energy_diffusion(u::AbstractArray; ν::Real=1.0)
    S = strain_field(u)
    ε = 2f0 * ν .* sum(S .^ 2, dims = (3, 4))  # sum over i,j
    ε = dropdims(ε, dims = (3, 4))
    return ε
end

function zero_crossing(y; direction=:both, eps=0.0)
    @assert direction in (:both, :rising, :falling)
    n = length(y)
    idx = Int[]
    # treat tiny values as exact zeros if eps>0
    yproc = copy(y)
    if eps > 0.0
        for i in eachindex(yproc)
            if abs(yproc[i]) <= eps
                yproc[i] = zero(yproc[i])
            end
        end
    end

    for i in 1:n-1
        a, b = yproc[i], yproc[i+1]
        if a*b < 0 || a == 0 || b == 0
            dir = if a < 0 && b > 0
                :rising
            elseif a > 0 && b < 0
                :falling
            elseif a == 0 && b != 0
                b > 0 ? :rising : :falling
            elseif b == 0 && a != 0
                a > 0 ? :falling : :rising 
            else
                # flat at zero
                nothing
            end
            if dir !== nothing && (direction == :both || dir == direction)
                push!(idx, i)
            end
        end
    end
    return idx
end

function train_force_plot(simdata::Any; train_idx=nothing, val_idx=nothing, test_idx=nothing)
    forces = simdata.force
    time = simdata.time
    drag = first.(forces)
    lift = last.(forces)

    zero_idxs = zero_crossing(lift; direction=:rising)
    

    plt = plot(time, [drag, lift],
        labels=["drag" "lift"],
        colors=[:red, :blue],
        xlabel="tU/L",
        ylabel="Pressure force coefficients",
        legend=:topright, 
        linewidth=1.5)


    if !isnothing(val_idx) && !isempty(val_idx)
        scatter!(plt, time[val_idx], lift[val_idx], 
        markersize = 2, color=:darkgreen, markerstrokewidth = 0, markershape =:dtriangle, 
        label="validation points")
    end
    # Highlight train/val region (before test starts)
    if !isnothing(train_idx) && !isempty(train_idx)
        train_range = first(train_idx) : last(train_idx)
        
        # Add vertical shaded region for train/val
        vspan!(plt, [time[first(train_range)], time[last(train_range)]];
            fillcolor=:green, alpha=0.1, label="train/val region")
        

    end
 
    # Highlight test region
    if !isnothing(test_idx) && !isempty(test_idx)
        test_range = first(test_idx) : last(test_idx)
        
        # Add vertical shaded region for test
        vspan!(plt, [time[first(test_range)], time[last(test_range)]];
            fillcolor=:purple, alpha=0.1, label="test region")
        
    end
    # Annotate zero crossings
    for (i, idx) in enumerate(zero_idxs)
        println(i % 2)
        shift = i % 2
        scatter!(plt, [time[idx]], [lift[idx]]; label=false, color=:black, markersize=3)
        annotate!(plt, time[idx], lift[idx] + 0.1 -(shift*0.2) , text(string(round(time[idx],digits = 3)), 8, :right))
    end

    # display(plt)
    return plt
end