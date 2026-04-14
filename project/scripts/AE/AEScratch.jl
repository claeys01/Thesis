using JLD2

AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"

checkpoint = JLD2.load(AE_path)
args_dict = checkpoint["args"]

# Display all arguments
for (k, v) in pairs(args_dict)
    println("$k: $v")
end