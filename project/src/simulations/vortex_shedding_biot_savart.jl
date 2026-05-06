function circle_shedding_biot(; Re=2500, U=1, n=2^8, m=2^8, mem=Array, perturb=true, Δt=0.25)

    radius = Float32(m / 16)                  # radius of the circle relative to the height of the domain
    center = (Float32(n / 4), Float32(m / 2)) # location of the circle relative to the height of the domain


    D = 2f0 * radius
    visc = U * D / Re        # <-- key change: ν ∝ 1/Re

    sdf(x, t) = √sum(abs2, x .- center) - radius
    sim = BiotSimulation(
        (n, m),
        (U, 0),          # flow velocity
        2f0radius;
        Δt=Δt,
        ν=visc, # defining viscosity
        body=AutoBody(sdf),
        mem=mem)
    perturb && perturb!(sim; noise=0.1)
    return sim
end

function run_sim(sim::AbstractSimulation; t_end=25, u₀=nothing, return_data=true, verbose=true, save_path=nothing)
    # Temporary storage while sampling
    time = Float32[]
    Δt = Float32[]
    u_list = Vector{Array{Float32,3}}()          # assuming u is 3D per snapshot
    p_list = Vector{Array{Float32,2}}()
    μ₀_list = Vector{Array{Float32,3}}()         # adjust if needed
    f_list = Vector{Array{Float32,3}}()
    force = Vector{Vector{Float32}}()
    ε = Float32[]

    # initializing simulation with desired flow field
    if !isnothing(u₀) && (size(sim.flow.u) == size(u₀))
        sim.flow.u .= u₀
    end

    while sim_time(sim) < t_end
        sim_step!(sim)
        verbose && sim_info(sim)
        if return_data
            scaled_force = get_forces(sim)
            push!(force, scaled_force)
            push!(u_list, copy(sim.flow.u))               # make a snapshot copy
            push!(p_list, copy(sim.flow.p))               # make a snapshot copy
            push!(μ₀_list, copy(sim.flow.μ₀))             # make a snapshot copy
            push!(f_list, copy(sim.flow.f))
            push!(time, Float32(round(sim_time(sim), digits=4)))
            push!(Δt, Float32(round(sim.flow.Δt[end], digits=3)))
        end
    end
    if return_data
        # Stack snapshots into dense arrays
        # u_list :: Vector{Array{T,3}} -> u :: Array{T,4}
        u = cat(u_list...; dims=4)  # (nx, ny, nchan, Nsnap)
        μ₀ = cat(μ₀_list...; dims=4)
        f = cat(f_list...; dims=4)
        p = cat(p_list...; dims=3)
        simdata = SimData(
            time=time,
            Δt=Δt,
            u=u,
            p=p,
            f=f,
            μ₀=μ₀,
            force=force,
            ε=ε,
            period_ranges=UnitRange{Int}[],
            reordered_ranges=UnitRange{Int}[],
            single_period_idx=1:0,
        )
        if !isnothing(save_path)
            @save save_path simdata
            @info "Saved simdata to $(save_path)"
        end
        return sim, simdata
    end
    return sim, nothing
end

# function get_forces!(sim,t)
#     sim_step!(sim,t,remeasure=false; verbose=true)
#     force = WaterLily.pressure_force(sim)
#     force./(0.5sim.L*sim.U^2) # scale the forces!
# end



