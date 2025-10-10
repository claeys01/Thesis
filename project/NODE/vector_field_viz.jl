using GLMakie
using OrdinaryDiffEq

A = [-0.1 2.0; -2.0 -0.1]
f(x, y) = A' * ([x, y].^3)

tmax = 100.0
xs = range(-2, 2; length=15)
ys = range(-2, 2; length=15)
ts = range(0, tmax; length=15)

# Create grid of positions
positions = [(x, t, y) for t in ts, y in ys, x in xs]
X = [pos[1] for pos in positions]
T = [pos[2] for pos in positions]
Z = [pos[3] for pos in positions]

# Compute vector field at each position
U = Float64[]
V = Float64[]
W = Float64[]
mag = Float64[]
for pos in positions
    x, t, y = pos
    vec = f(x, y)
    push!(U, vec[1])
    push!(V, 0.0)      # No time component in direction
    push!(W, vec[2])
    push!(mag, sqrt(vec[1]^2 + vec[2]^2) + eps())
end

# Normalize U and W
U ./= mag
W ./= mag


fig = Figure()
ax  = Axis3(fig[1,1], xlabel="u₁", ylabel="time", zlabel="u₂",
            title="3D Vector field: ẋ = A'*(u.^3)")

arrows3d!(ax, X, T, Z, U, V, W, lengthscale=0.2, color=mag, alpha=0.5)

using OrdinaryDiffEq

function trueODEfunc(du, u, p, t)
    A = [-0.1 2.0; -2.0 -0.1]
    du .= (A' * (u .^ 3))
end

prob = ODEProblem(trueODEfunc, [1.0, 0.2], (0.0, tmax))
sol  = solve(prob, Tsit5())

# Plot the trajectory in 3D: (u₁, time, u₂)
lines!(ax, sol[1, :], sol.t, sol[2, :], color=:black, linewidth=2)

display(fig)