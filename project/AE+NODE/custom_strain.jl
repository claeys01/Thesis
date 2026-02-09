using WaterLily
using WaterLily: δ, CI, inside, ∂
using Plots
using Statistics

# Generate a vector field u = (u, v) where the velocity gradient tensor is constant
# 
# Velocity gradient tensor: ∇u = [∂u/∂x  ∂u/∂y]
#                                [∂v/∂x  ∂v/∂y]
#
# For a linear velocity field: u(x,y) = a*x + b*y + c
#                              v(x,y) = d*x + e*y + f
# The gradient is constant:    ∇u = [a  b]
#                                   [d  e]

# # Define the constant gradient components
# a = 1.0   # ∂u/∂x
# b = 0.5   # ∂u/∂y  
# d = -0.5  # ∂v/∂x
# e = -1.0  # ∂v/∂y (note: a + e = 0 means divergence-free!)

# # Grid size
# nx, ny = 20, 20

# # Create coordinate arrays
# x = range(0, nx-1, length=nx)
# y = range(0, ny-1, length=ny)

# # Generate the velocity field
# u_field = zeros(Float64, nx, ny, 2)

# for j in 1:ny
#     for i in 1:nx
#         u_field[i, j, 1] = a * x[i] + b * y[j]  # u-component
#         u_field[i, j, 2] = d * x[i] + e * y[j]  # v-component
#     end
# end

# # Verify the gradients using finite differences
# println("Expected gradient tensor:")
# println("  ∂u/∂x = $a")
# println("  ∂u/∂y = $b")
# println("  ∂v/∂x = $d")
# println("  ∂v/∂y = $e")
# println("  Divergence (∂u/∂x + ∂v/∂y) = $(a + e)")

# # Compute numerical gradients at interior points
# I = CartesianIndex(10, 10)
# # I = CartesianIndex(19, 19)
# dx = x[2] - x[1]
# dy = y[2] - y[1]

# # @show I, δ(1,I), I + δ(1,I)

# dudx_num = (u_field[I + δ(1,I), 1] - u_field[I - δ(1,I), 1]) / (2*dx)
# dudy_num = (u_field[I + δ(2,I), 1] - u_field[I - δ(2,I), 1]) / (2*dy)
# dvdx_num = (u_field[I + δ(1,I), 2] - u_field[I - δ(1,I), 2]) / (2*dx)
# dvdy_num = (u_field[I + δ(2,I), 2] - u_field[I - δ(2,I), 2]) / (2*dy)

# println("\nNumerical gradients at I=$I:")
# println("  ∂u/∂x ≈ $dudx_num (expected: $a)")
# println("  ∂u/∂y ≈ $dudy_num (expected: $b)")
# println("  ∂v/∂x ≈ $dvdx_num (expected: $d)")
# println("  ∂v/∂y ≈ $dvdy_num (expected: $e)")

# dudx_wat = ∂(1, 1, I, u_field)
# dudy_wat = ∂(1, 2, I, u_field)
# dvdx_wat = ∂(2, 1, I, u_field)
# dvdy_wat = ∂(2, 2, I, u_field)

# println("\nNumerical (WaterLily) gradients at I=$I:")
# println("  ∂u/∂x ≈ $dudx_wat (expected: $a)")
# println("  ∂u/∂y ≈ $dudy_wat (expected: $b)")
# println("  ∂v/∂x ≈ $dvdx_wat (expected: $d)")
# println("  ∂v/∂y ≈ $dvdy_wat (expected: $e)")

# # Visualize the vector field
# X = [x[i] for i in 1:nx, j in 1:ny]
# Y = [y[j] for i in 1:nx, j in 1:ny]
# U = u_field[:, :, 1]
# V = u_field[:, :, 2]

# p = quiver(vec(X[1:2:end, 1:2:end]), vec(Y[1:2:end, 1:2:end]), 
#            quiver=(vec(U[1:2:end, 1:2:end])*0.03, vec(V[1:2:end, 1:2:end])*0.03),
#            aspect_ratio=1, 
#            title="Linear velocity field with constant gradient\n∇·u = $(a+e)",
#            xlabel="x", ylabel="y",
#            legend=false)
# # display(p)
# nothing

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

using Revise
includet("../simulations/vortex_shedding_biot_savart.jl")
includet("../custom.jl")
includet("../AE/Lux_AE.jl")

simdata = load_simdata("data/datasets/RE2500/2e8/U_128_full.jld2")
u = simdata.u[:, :, :, 1]

# dudx_interior, dudy, dvdx, dvdy_interior = velocity_gradient_vectorized(sim.flow.u)
# @show size(dudx_interior), size(dudy), size(dvdx), size(dvdy_interior)
# @show mean(dudx_interior), mean(dudy), mean(dvdx), mean(dvdy_interior)

div = div_vectorized(u)
div_wat = div_field(u)
@show mean(div), size(div)
@show mean(div_wat), size(div_wat)
@show maximum(abs.(div .- div_wat))  # Should be ~0 (machine precision)

S_mat = strain_field(u)
@show mean(S_mat), size(S_mat)
S_new = zeros(256,256,2,2)

S11, S12, S22 = strain_rate_vectorized(u)

S_new[:,:,1,1], S_new[:,:,1,2], S_new[:,:,2,1], S_new[:,:,2,2] = S11, S12, S12, S22
@show mean(S_new), size(S_new)
@show maximum(abs.(S_mat .- S_new))  # Should be ~0 (machine precision)



# nothing