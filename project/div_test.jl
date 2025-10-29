using Revise
using WaterLily: inside
using JLD2
using Statistics

includet("custom.jl")   # provides grad, RHS, get_random_snapshots, downsample_RHS_data!, remove_ghosts
includet("simulations/vortex_shedding_biot_savart.jl")

@load "data/datasets/RHS_biot_data_arr_force.jld2" RHS_data

idx = div(1795, 2)
println("Selected snapshot index: $idx\n")

flow = RHS_data["flow"][idx]
RHS_field  = RHS_data["RHS"][idx]

u = flow.u

divergence_ad(U; dx=1.0, dy=1.0) = begin
    ux = @view U[:,:,1]
    uy = @view U[:,:,2]
    dudx = (circshift(ux, (-1,0)) .- circshift(ux, (1,0))) ./ (2*dx)
    dvdy = (circshift(uy, (0,-1)) .- circshift(uy, (0,1))) ./ (2*dy)
    dudx .+ dvdy
end

# create an array of zeros with the same element type and shape as flow.p
jemoeder = Matrix(zeros(eltype(flow.p), size(flow.p)))

# @inside jemoeder[I] = WaterLily.div(I, u)
@inside jemoeder[I] = WaterLily.div(I, u)


# Now do the same operation for the RHS matrix

jemoeder_RHS = Matrix(zeros(eltype(flow.p), size(flow.p)))

@inside jemoeder_RHS[I] = WaterLily.div(I, RHS_field)


println("velocity u: size = $(size(u)), eltype = $(eltype(u)), typeof = $(typeof(u))")
println("pressure field flow.p: size = $(size(flow.p)), eltype = $(eltype(flow.p)), typeof = $(typeof(flow.p))\n")
println("Created jemoeder (for div(u)): size = $(size(jemoeder)), eltype = $(eltype(jemoeder)))")
println("jemoeder (div u) stats: mean = $(mean(jemoeder)), min = $(minimum(jemoeder)), max = $(maximum(jemoeder))\n")
println("RHS object: size = $(size(RHS_field)), eltype = $(eltype(RHS_field)), typeof = $(typeof(RHS_field))")
println("Created jemoeder_RHS (for div(RHS)): size = $(size(jemoeder_RHS)), eltype = $(eltype(jemoeder_RHS)))")
println("jemoeder_RHS (div RHS) stats: mean = $(mean(jemoeder_RHS)), min = $(minimum(jemoeder_RHS)), max = $(maximum(jemoeder_RHS))")

mean(divergence_ad(u))
mean(divergence_ad(flow.p))

