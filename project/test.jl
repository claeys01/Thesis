using JLD2
using Revise

includet("custom.jl")

@load "data/datasets/128_RHS_biot_data_arr_force_period.jld2" data

preprocess_data!(data; clip_bc=true)

