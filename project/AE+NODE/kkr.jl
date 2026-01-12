using WaterLily
using WaterLily: @loop, S, inside, inside_u, BC!
using Random
using BiotSavartBCs

# reproducible randomness
Random.seed!(1234)

include("../custom.jl")

n = 2^5
ng = 2
domsize = (n+ng, n+ng, 2)

mat = randn(Float64, domsize...)
cr = 2:(n-1)
@show typeof(cr)
mat2 = deepcopy(mat[cr,cr, :])
@show size(mat), size(mat2)2
Sfield = zeros(Float64, domsize..., 2)
Sfield2 = zeros(Float64, size(mat2)..., 2)


@show inside_u(mat), CartesianIndices((1:n, 1:n))
@assert typeof(inside_u(mat)) == typeof(CartesianIndices((1:n, 1:n)))

@loop Sfield[I,:,:] .= S(I, mat) over I ∈ inside_u(mat)
@loop Sfield2[J,:,:] .= S(J, mat2) over J ∈ CartesianIndices((1:n, 1:n))

Sfield = Sfield[cr, cr, :, :]

@assert size(Sfield) == size(Sfield2)
@show mean(Sfield), mean(Sfield2)

# # Svalidate = strain_field(mat)
# @assert Sfield == Sfield2

function pad_edges(A::AbstractArray{T,3}, padval::T) where {T}
    Nx, Ny, C = size(A)
    B = fill(padval, Nx + 2, Ny + 2, C)          # border already set
    @views B[2:Nx+1, 2:Ny+1, :] .= A             # copy interior
    return B
end


function apply_biot_savart_bcs!(u::AbstractArray{T,3}; Δ::Real=1.0, θ::Real=0.5) where {T<:Real}
    Nx, Ny, C = size(u)
    @assert C == 2 "u must be (Nx, Ny, 2)"
    @assert Nx ≥ 3 && Ny ≥ 3

    # interior velocity views (no ghosts)
    nx, ny = Nx - 2, Ny - 2
    @views ux = u[2:end-1, 2:end-1, 1]
    @views uy = u[2:end-1, 2:end-1, 2]

    # scalar vorticity ω = ∂v/∂x - ∂u/∂y on interior cell centers
    ω = zeros(eltype(u), nx, ny)
    @inbounds for j in 2:ny-1, i in 2:nx-1
        ∂v∂x = (uy[i+1, j] - uy[i-1, j]) / (2Δ)
        ∂u∂y = (ux[i, j+1] - ux[i, j-1]) / (2Δ)
        ω[i, j] = ∂v∂x - ∂u∂y
    end

    # interior coordinates (cell centers)
    x = (0:nx-1) .* Δ .+ Δ/2
    y = (0:ny-1) .* Δ .+ Δ/2

    # Build Biot–Savart FMM tree from interior vorticity
    # (Check deps/BiotSavartBCs/examples for exact ctor names if needed)
    tree = BiotSavartBCs.Tree(x, y, ω; θ=θ)

    # Boundary sample points (centered on ghost faces)
    Lx, Ly = nx * Δ, ny * Δ
    xb_left,  yb_left  = fill(-Δ/2, ny), y
    xb_right, yb_right = fill(Lx + Δ/2, ny), y
    xb_bot,   yb_bot   = x, fill(-Δ/2, nx)
    xb_top,   yb_top   = x, fill(Ly + Δ/2, nx)

    # Induced velocities at boundary points
    ulx, uly = BiotSavartBCs.velocity(tree, xb_left,  yb_left)
    urx, ury = BiotSavartBCs.velocity(tree, xb_right, yb_right)
    ubx, uby = BiotSavartBCs.velocity(tree, xb_bot,   yb_bot)
    utx, uty = BiotSavartBCs.velocity(tree, xb_top,   yb_top)

    # Write ghost cells: left/right columns, bottom/top rows
    @views begin
        u[1,        2:end-1, 1] .= ulx;  u[1,        2:end-1, 2] .= uly   # left
        u[end,      2:end-1, 1] .= urx;  u[end,      2:end-1, 2] .= ury   # right
        u[2:end-1,  1,        1] .= ubx; u[2:end-1,  1,        2] .= uby  # bottom
        u[2:end-1,  end,      1] .= utx; u[2:end-1,  end,      2] .= uty  # top
    end
    return u
end