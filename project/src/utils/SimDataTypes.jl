module SimDataTypes

export SimData
export EpochData
export LatentData

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

Base.@kwdef struct EpochData
    Xin::Array{Float32,4}
    Xout::Array{Float32,4}
    μ₀::Array{Float32,4}
end

Base.@kwdef mutable struct LatentData
    z::Matrix{Float32}
    time::Vector{Float32}
end


end 