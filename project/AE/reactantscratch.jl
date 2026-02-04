using Reactant
using Random

device = reactant_device(; force=true)    
cpu = cpu_device()

function foo(a, b)
    return a .+ b
end

a = randn(10,10,2,10) |> dev
b = randn(10,10,2,10) |> dev

@compile foo(a, b)

out = foo(a,b)
@show typeof(out)

out_arr = Array(out)
@show typeof(out_arr)
