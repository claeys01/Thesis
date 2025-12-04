module SimDataTypes

export SimData

Base.@kwdef mutable struct SimData
    time::Vector{Float32}
    Δt::Vector{Float32}
    u::Array{Float32,4}
    μ₀::Array{Float32,4}
    force::Vector{Vector{Float32}}
    ε::Vector{Float32}
    period_ranges::Vector{UnitRange{Int}}
    reordered_ranges::Vector{UnitRange{Int}}
    single_period_idx::UnitRange{Int}
end

end # module

    # time::Vector{Float32}
    # Δt::Vector{Float32}
    # u::Array{Float32,4}                  # (H, W, C, T)
    # μ₀::Array{Float32,2}                 # (nμ, T)
    # force::Vector{NTuple{2,Float32}}     # (Cd, Cl)
    # ε::Vector{Float32}                   # dissipation per snapshot
    # period_ranges::Vector{UnitRange{Int}}
    # reordered_ranges::Vector{UnitRange{Int}}
    # single_period_idx::UnitRange{Int}