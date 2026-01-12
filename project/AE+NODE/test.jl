using PaddedViews
using Random
using WaterLily

Random.seed!(42)

A = round.(randn(Float32, 4, 4, 2); digits=3)
@show A

"""
Pad a (Nx, Ny, C) array with a 1-cell border filled with `padval`.
Returns an (Nx+2, Ny+2, C) array.
"""
function pad_edges(A::AbstractArray{T,3}, padval::T) where {T}
    Nx, Ny, C = size(A)
    B = fill(padval, Nx + 2, Ny + 2, C)          # border already set
    @views B[2:Nx+1, 2:Ny+1, :] .= A             # copy interior
    return B
end

# A = rand(Float32, 256, 256, 2)
B = pad_edges(A, -1f0)   # border filled with 0.0f0
size(B)                 # (258, 258, 2)
@show typeof(A)


# A, B

