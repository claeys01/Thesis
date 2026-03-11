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
    RHS = conv_diff!(flow.f,flow.u⁰,flow.σ,λ;ν=flow.ν,perdir=flow.perdir) - grad_p(flow)
    return RHS
end

function remove_ghosts(snapshot::AbstractArray)
    return snapshot[2:end-1, 2:end-1, :, :]
end

function preprocess_data!(data::SimData;
                          verbose::Bool = true)


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


function strain_field(u::AbstractArray{T,N}; buff::Int64=1) where {T,N}
    if N == 3
        H, W, C = size(u)
        S_tensor = zeros(T, H, W, C, C)
        scalar_mat = zeros(T, H, W)
        @loop S_tensor[I, :, :] .= S(I, u) over I ∈ inside(scalar_mat; buff=buff)
        return remove_buff(S_tensor, buff)
    elseif N == 4
        H, W, C, t = size(u)
        S_tensor = zeros(T, H-2*buff, W-2*buff, C, C, t)
        for i in 1:t
            S_tensor[:, :, :, :, i] .= strain_field(u[: ,: ,: ,i]; buff=buff)
        end
        return S_tensor
    else
        throw(ArgumentError("strain_field expects a 3- or 4-dimensional array"))
    end
end


function kinetic_energy_dissipation(u::AbstractArray{T,N}; ν::Real=1.0, avg::Bool=false, buff::Int64=1) where {T, N}
    S = strain_field(u; buff=buff)
    ε = 2 * ν .* sum(S .^ 2, dims = (3, 4))  # sum over i,j
    ε = dropdims(ε, dims = (3, 4))
    @show size(ε)
    avg ? vec(dropdims(mean(ε; dims=(1,2)); dims=(1,2))) :  ε
end

function remove_buff(arr::AbstractArray{T, N}, buff::Int) where {T,N}
    if N == 2
        return arr[1+buff:end-buff, 1+buff:end-buff]
    elseif N == 4
        return arr[1+buff:end-buff, 1+buff:end-buff, :, :]
    else
        @error "remove buff for $N dimensions not implemented"
    end
end

function div_field(u::AbstractArray{T,N}; avg=false, buff=1) where {T,N}
    if N == 3
        H, W, _ = size(u)
        div_mat = zeros(T, H, W)
        @loop div_mat[I] = WaterLily.div(I,u) over I ∈ WaterLily.inside(div_mat; buff=buff)
        return remove_buff(div_mat, buff)
        # return div_mat
    elseif N == 4
        H, W, _, t = size(u)
        div_field_arr = zeros(T, H-2*buff, W-2*buff, t)
        for i in 1:t
            div_field_arr[:, :, i] = div_field(u[:, :, :, i]; buff=buff)
        end
        # return div_field_arr
        avg ? (return vec(dropdims(mean(div_field_arr; dims=(1,2)); dims=(1,2)))) : (return div_field_arr)
    else
        throw(ArgumentError("div_field expects a 3- or 4-dimensional array"))
    end
end



# replacing flow field with simulations
function insert_prediction!(sim::AbstractSimulation, û::AbstractArray{T,3}) where {T}
    sim_size = size(sim.flow.u); pred_size = size(û)
    @assert pred_size == (sim_size[1]-2, sim_size[2]-2, sim_size[3]) "simulation and prediction sizes do not match"
    sim.flow.u[2:end-1, 2:end-1, :] .= û
    return sim
end

function custom_BDIM!(a::Flow, dt)
    # dt = a.Δt[end]

    @loop a.f[Ii] = a.u⁰[Ii]+dt*a.f[Ii]-a.V[Ii] over Ii in CartesianIndices(a.f)
    @loop a.u[Ii] += WaterLily.μddn(Ii,a.μ₁,a.f)+a.V[Ii]+a.μ₀[Ii]*a.f[Ii] over Ii ∈ inside_u(size(a.p))
end

function custom_biot_project!(a::Flow{n},ml_b::MultiLevelPoisson,ω,x₀,tar,ftar,U;fmm=true,w=1,tol=1e-4,itmx=32) where n
    dt = w*a.Δt[end]; a.p .*= dt  # Scale p *= w*Δt
    # dt *= w; a.p .*= dt  # Scale p *= w*Δt
    BiotSavartBCs.apply_grad_p!(a.u,ω,a.p,a.μ₀) # Apply u-=μ₀∇p & ω=∇×u
    x₀ .= a.p; fill!(a.p,0)       # x₀ holds p solution
    BiotSavartBCs.biotBC!(a.u,U,ω,tar,ftar;fmm) # Apply domain BCs
    # Set residual
    b = ml_b.levels[1]; b.r .= 0
    @inside b.r[I] = ifelse(b.iD[I]==0,0,WaterLily.div(I,a.u))
    BiotSavartBCs.fix_resid!(b.r,a.u,tar[1]) # only fix on the boundaries
    nᵖ,nᵇ,r₂ = 0,0,L₂(b)
    @log ", $nᵖ, $(WaterLily.L∞(b)), $r₂, $nᵇ\n"
    while nᵖ<itmx
        rtol = max(tol,0.1r₂)
        while nᵖ<itmx
            WaterLily.Vcycle!(ml_b); WaterLily.smooth!(b)
            r₂ = L₂(b); nᵖ+=1
            r₂<rtol && break
        end
        BiotSavartBCs.apply_grad_p!(a.u,ω,a.p,a.μ₀)   # Update u,ω
        x₀ .+= a.p; fill!(a.p,0)        # Update solution
        BiotSavartBCs.biotBC_r!(b.r,a.u,U,ω,tar,ftar;fmm) # Update BC+residual
        r₂ = L₂(b); nᵇ+=1
        @log ", $nᵖ, $(WaterLily.L∞(b)), $r₂, $nᵇ\n"
        r₂<tol && break
    end
    push!(ml_b.n,nᵖ)
    BiotSavartBCs.pflowBC!(a.u)  # Update ghost BCs (domain is already correct)
    a.p .= x₀/dt   # copy-scaled pressure solution
end


"""
Imposing Biot Savart BCs on a simulation, and also filling the pressure field.
Function assumes that the pressure field is filled, 
Use for imposing BCs and pressure field on predicted flow field
"""
function impose_biot_bc!(a::Flow{N}, b, ω...;λ=quick, fmm=true) where {N}

    t₁ = sum(a.Δt)
    U = BiotSavartBCs.BCTuple(a.uBC, t₁, N) 

    a.u⁰ .= a.u; WaterLily.scale_u!(a,0)
    conv_diff!(a.f,a.u⁰,a.σ,λ,ν=a.ν)
    WaterLily.BDIM!(a);
    custom_biot_project!(a,b,ω...,U;fmm) # new

    WaterLily.conv_diff!(a.f,a.u,a.σ,λ,ν=a.ν)
    WaterLily.BDIM!(a); WaterLily.scale_u!(a,0.5)
    custom_biot_project!(a,b,ω...,U;fmm,w=0.5) # new
    # push!(a.Δt,WaterLily.CFL(a))


    # WaterLily.measure!(a,body;t₁,ϵ=1)
    WaterLily.update!(b)

    a.u⁰ .= a.u; WaterLily.scale_u!(a,0)
    conv_diff!(a.f,a.u⁰,a.σ,λ,ν=a.ν)
    WaterLily.BDIM!(a);
    custom_biot_project!(a,b,ω...,U;fmm) # new

    WaterLily.conv_diff!(a.f,a.u,a.σ,λ,ν=a.ν)
    WaterLily.BDIM!(a); WaterLily.scale_u!(a,0.5)
    custom_biot_project!(a,b,ω...,U;fmm,w=0.5) # new
    # push!(a.Δt,WaterLily.CFL(a))
    # a.u .= a.u⁰
end

impose_biot_bc!(sim::BiotSimulation) = impose_biot_bc!(sim.flow, sim.pois, sim.ω, sim.x₀,sim.tar,sim.ftar;fmm=sim.fmm)  

function impose_biot_bc_on_snapshot(û::AbstractArray{T,N}; return_sim=false) where {T, N}
    if N == 3
        sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)
        insert_prediction!(sim, û)
        impose_biot_bc!(sim)
        if return_sim
            return sim
        else
            return sim.flow.u
        end
    elseif N == 4
        return_sim && @warn "returning an array of BiotSimulations, might not be usefull"
        H, W, C, t = size(û)
        snapshot_arr = zeros(T, H+2, W+2, C, t)
        for i in 1:t
            snapshot_arr[:, :, :, i] =  impose_biot_bc_on_snapshot(û[:, :, :, i])
        end
        return snapshot_arr
    else
        throw(ArgumentError("impose_biot_bc_on_snapshot expects a 3- or 4-dimensional array"))
    end
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

function train_force_plot(forces::Vector{Vector{Float32}}, time::Vector{Float32}; train_idx=nothing, val_idx=nothing, test_idx=nothing, show_zeros=true)
    drag = first.(forces)
    lift = last.(forces)
    zero_idxs = zero_crossing(lift; direction=:rising)

    plt = plot(xlabel="tU/L",
        ylabel="Pressure force coefficients",
        legend=:topright) 
    plot!(plt, time, drag, label="drag", color=:red, linewidth=1.5)
    plot!(plt, time, lift, label="lift", color=:blue, linewidth=1.5)


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
    if show_zeros
        for (i, idx) in enumerate(zero_idxs)
            shift = i % 2
            scatter!(plt, [time[idx]], [lift[idx]]; label=false, color=:black, markersize=3)
            annotate!(plt, time[idx], lift[idx] + 0.1 -(shift*0.2) , text(string(round(time[idx],digits = 3)), 8, :right))
        end
    end

    # display(plt)
    return plt
end

function train_force_plot(simdata::SimData; 
        train_idx=nothing, val_idx=nothing, test_idx=nothing, show_zeros=true)
   train_force_plot(simdata.force, simdata.time; 
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, show_zeros=show_zeros)
end

function velocity_gradient_vectorized(u::AbstractArray{T,3}; buff=1) where T
    # Using central differences for cross-terms (like WaterLily's ∂(i,j,I,u) for i≠j)
    # and one-sided differences for inline terms (like WaterLily's ∂(i,I,u))
    H, W, _, = size(u)

    i_range = (1+buff):(H-buff)
    j_range = (1+buff):(W-buff)

    # Inline terms: ∂uᵢ/∂xᵢ uses one-sided difference: u[I+δ(i),i] - u[I,i]
    # ∂u/∂x: u[i+1,j,1] - u[i,j,1]
    dudx = u[i_range .+ 1, j_range, 1] .- u[i_range, j_range, 1]    

    # ∂v/∂y: u[i,j+1,2] - u[i,j,2]  
    dvdy = u[i_range, j_range .+ 1, 2] .- u[i_range, j_range, 2]

    # Cross terms: ∂uᵢ/∂xⱼ (i≠j) uses central difference / 4 (WaterLily convention)
    # ∂u/∂y: (u[I+δy] + u[I+δy+δx] - u[I-δy] - u[I-δy+δx]) / 4
    #      = (u[i,j+1,1] + u[i+1,j+1,1] - u[i,j-1,1] - u[i+1,j-1,1])/4
    dudy = (u[i_range, j_range .+ 1, 1] .+ u[i_range .+ 1, j_range .+ 1, 1]
          .- u[i_range, j_range .- 1, 1] .- u[i_range .+ 1, j_range .- 1, 1]) ./ 4

    
    # ∂v/∂x: (u[I+δx] + u[I+δx+δy] - u[I-δx] - u[I-δx+δy]) / 4
    #      = (u[i+1,j,2] + u[i+1,j+1,2] - u[i-1,j,2] - u[i-1,j+1,2])/4
    dvdx = (u[i_range .+ 1, j_range, 2] .+ u[i_range .+ 1, j_range .+ 1, 2]
          .- u[i_range .- 1, j_range, 2] .- u[i_range .- 1, j_range .+ 1, 2]) ./ 4    
    return dudx, dudy, dvdx, dvdy
end

function velocity_gradient_vectorized(u::AbstractArray{T,4}; buff=1) where T
    H, W, _, B = size(u)
    
    i_range = (1+buff):(H-buff)
    j_range = (1+buff):(W-buff)
    
    # Diagonal terms
    dudx = u[i_range .+ 1, j_range, 1, :] .- u[i_range, j_range, 1, :]
    dvdy = u[i_range, j_range .+ 1, 2, :] .- u[i_range, j_range, 2, :]
    
    # Off-diagonal terms (4-point stencil)
    dudy = (u[i_range, j_range .+ 1, 1, :] .+ u[i_range .+ 1, j_range .+ 1, 1, :]
          .- u[i_range, j_range .- 1, 1, :] .- u[i_range .+ 1, j_range .- 1, 1, :]) ./ 4
    
    dvdx = (u[i_range .+ 1, j_range, 2, :] .+ u[i_range .+ 1, j_range .+ 1, 2, :]
          .- u[i_range .- 1, j_range, 2, :] .- u[i_range .- 1, j_range .+ 1, 2, :]) ./ 4
    
    return dudx, dudy, dvdx, dvdy
end


function div_vectorized(u::AbstractArray{T,3}; buff=1) where T
    H, W, _ = size(u)

    i_range = (1+buff):(H-buff)
    j_range = (1+buff):(W-buff)
    
    dudx = u[i_range .+ 1, j_range, 1] .- u[i_range, j_range, 1]
    dvdy = u[i_range, j_range .+ 1, 2] .- u[i_range, j_range, 2]
    
    return dudx .+ dvdy
end

# Batched version
function div_vectorized(u::AbstractArray{T,4}; buff=1) where T
    H, W, _, B = size(u)
    
    i_range = (1+buff):(H-buff)
    j_range = (1+buff):(W-buff)
    
    dudx = u[i_range .+ 1, j_range, 1, :] .- u[i_range, j_range, 1, :]
    dvdy = u[i_range, j_range .+ 1, 2, :] .- u[i_range, j_range, 2, :]
    
    return dudx .+ dvdy
end

function curl_vectorized(u::AbstractArray{T,3}; buff=1) where T
    H, W, _ = size(u)
    i_range = (1+buff):(H-buff)
    j_range = (1+buff):(W-buff)
    
    # ∂v/∂x (4-point stencil)
    dvdx = (u[i_range .+ 1, j_range, 2] .+ u[i_range .+ 1, j_range .+ 1, 2]
          .- u[i_range .- 1, j_range, 2] .- u[i_range .- 1, j_range .+ 1, 2]) ./ 4
    
    # ∂u/∂y (4-point stencil)
    dudy = (u[i_range, j_range .+ 1, 1] .+ u[i_range .+ 1, j_range .+ 1, 1]
          .- u[i_range, j_range .- 1, 1] .- u[i_range .+ 1, j_range .- 1, 1]) ./ 4
    
    return dvdx .- dudy
end

function curl_vectorized(u::AbstractArray{T,4}; buff=1) where T
    H, W, _, B = size(u)
    i_range = (1+buff):(H-buff)
    j_range = (1+buff):(W-buff)
    
    dvdx = (u[i_range .+ 1, j_range, 2, :] .+ u[i_range .+ 1, j_range .+ 1, 2, :]
          .- u[i_range .- 1, j_range, 2, :] .- u[i_range .- 1, j_range .+ 1, 2, :]) ./ 4
    
    dudy = (u[i_range, j_range .+ 1, 1, :] .+ u[i_range .+ 1, j_range .+ 1, 1, :]
          .- u[i_range, j_range .- 1, 1, :] .- u[i_range .+ 1, j_range .- 1, 1, :]) ./ 4
    
    return dvdx .- dudy
end

function strain_rate_vectorized(u::AbstractArray{T,3}; buff=1) where T
    dudx, dudy, dvdx, dvdy = velocity_gradient_vectorized(u; buff=buff)
    
    S11 = dudx                      # ∂u/∂x
    S22 = dvdy                      # ∂v/∂y
    S12 = (dudy .+ dvdx) ./ 2       # 0.5*(∂u/∂y + ∂v/∂x)
    
    return S11, S12, S22
end

function strain_rate_vectorized(u::AbstractArray{T,4}; buff=1) where T
    dudx, dudy, dvdx, dvdy = velocity_gradient_vectorized(u; buff=buff)
    
    S11 = dudx
    S22 = dvdy
    S12 = (dudy .+ dvdx) ./ 2
    
    return S11, S12, S22
end

function rotation_rate_vectorized(u::AbstractArray{T,3}; buff=1) where T
    dudx, dudy, dvdx, dvdy = velocity_gradient_vectorized(u; buff=buff)
    
    # Ω12 = 0.5*(∂u/∂y - ∂v/∂x) = -Ω21
    Ω12 = (dudy .- dvdx) ./ 2
    return Ω12
end

function rotation_rate_vectorized(u::AbstractArray{T,4}; buff=1) where T
    dudx, dudy, dvdx, dvdy = velocity_gradient_vectorized(u; buff=buff)
    Ω12 = (dudy .- dvdx) ./ 2
    return Ω12
end


function RST(u::AbstractArray{T, 4}, μ₀::AbstractArray{T, 3}) where {T}

    ū = mean(u[:, :, 1, :])
    v̄ = mean(u[:, :, 2, :])
    
    u′ = (u[:, :, 1, :] .- ū) .* μ₀[:, :, 1]                  # broadcast over time
    v′ = (u[:, :, 2, :] .- v̄) .* μ₀[:, :, 2] 

    # Reynolds stresses (still (Nx,Ny,1); you can drop dim 3 if you want)
    uu = dropdims(mean(u′ .^ 2;  dims=3); dims=3) # ⟨u'u'⟩
    vv = dropdims(mean(v′ .^ 2;  dims=3); dims=3) # ⟨v'v'⟩
    uv = dropdims(mean(u′ .* v′; dims=3); dims=3) # ⟨u'v'⟩
    return (uu, vv, uv)
end

function plot_reynolds_stresses(uu, vv, uv)
    # Compute clims for each component: (min, max) centered around mean
    # uu_min, uu_max = (0.0, maximum(uu))
    # vv_min, vv_max = (0.0, maximum(vv))
    uu_min, uu_max = extrema(uu)
    vv_min, vv_max = extrema(vv)
    uv_min, uv_max = extrema(uv)
    # uu_min, uu_max = 0.0, mean(uu) + std(uu)
    # vv_min, vv_max = 0.0, mean(vv) + std(vv)
    # uv_min, uv_max = mean(uv) - std(uv), mean(uv) + std(uv)
    
    
    # Plot ⟨u'u'⟩
    plt_uu = flood(uu; 
        clims=(uu_min, uu_max),
        levels=20,
        color=:viridis,
        title="⟨u'u'⟩",
        xlabel="x",
        ylabel="y",
        aspectratio=:equal, 
        border=:none, 
        framestyle=:none,
        axis=nothing
    )
    
    # Plot ⟨v'v'⟩
    plt_vv = flood(vv;
        clims=(vv_min, vv_max),
        levels=20,
        color=:viridis,
        title="⟨v'v'⟩",
        xlabel="x",
        ylabel="y",
        aspectratio=:equal,
        border=:none, 
        framestyle=:none,
        axis=nothing
    )
    
    # Plot ⟨u'v'⟩
    plt_uv = flood(uv;
        clims=(uv_min, uv_max),
        levels=20,
        color=:viridis,
        title="⟨u'v'⟩",
        xlabel="x",
        ylabel="y",
        aspectratio=:equal, 
        border=:none, 
        framestyle=:none,
        axis=nothing
    )
    
    # Combine into a single figure
    plt_combined = plot(plt_uu, plt_vv, plt_uv;
        layout=(1, 3),
        size=(1200, 350),
        dpi=150,
        # plot_title="Reynolds Stress Components (Re=2500)",
        # plot_titlefontsize=12
    )
    
    return plt_combined, (plt_uu, plt_vv, plt_uv)
end
plot_reynolds_stresses(simdata::SimData) = plot_reynolds_stresses(RST(simdata.u, simdata.μ₀[:, :, :, 1])...)
