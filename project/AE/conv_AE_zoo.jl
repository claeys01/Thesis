using JLD2
using MLUtils: DataLoader
includet("../custom.jl")


function get_data(batch_size; tmin=-1, tmax=-1, n_samples=500)
    @load "/home/matth/Thesis/data/RHS_shedding_data_arr.jld2" RHS_data
    downsample_RHS_data!(RHS_data; tmin=tmin, tmax=tmax, n_samples=n_samples)
    X = cat(RHS_data["RHS"]...; dims=4)   # now X has shape (H,W,C,N)
    return DataLoader(X, batchsize=batch_size, shuffle=true)
end
