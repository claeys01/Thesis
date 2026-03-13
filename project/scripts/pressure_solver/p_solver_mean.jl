using Thesis
using WaterLily
using Statistics
using Plots
using TimerOutputs
using BiotSavartBCs
using Printf

import WaterLily: Vcycle!,smooth!, scale_u!, conv_diff!, udf!, accelerate!, BDIM!
import BiotSavartBCs: apply_grad_p!, biotBC!, fix_resid!, biotBC_r!, pflowBC!, BCTuple

sim = circle_shedding_biot(; mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)

# Load data
node_path = "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)
simdata = load_simdata(aenode.ae_args.full_data_path)

# Load snapshot at index 1
random_int = 1
u, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]


function get_forces(sim::BiotSimulation)
    raw_force = WaterLily.pressure_force(sim)
    scaled_force = Float32.(raw_force ./ (0.5 * sim.L * sim.U^2))
    return scaled_force
end

# # Set initial condition
sim.flow.u .= u

