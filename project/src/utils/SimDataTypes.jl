module SimDataTypes

export SimData
export EpochData
export LatentData

Base.@kwdef mutable struct SimData
    time::Vector{Float32}
    Δt::Vector{Float32}
    u::Array{Float32,4}
    p::Array{Float32,3}
    μ₀::Array{Float32,4}
    force::Vector{Vector{Float32}}
    chunk_ranges::Vector{UnitRange{Int}} = UnitRange{Int}[]
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