using Thesis
using WaterLily
using Statistics
using Plots
using TimerOutputs
using BiotSavartBCs
using Printf

import WaterLily: Vcycle!,smooth!, scale_u!
import BiotSavartBCs: apply_grad_p!, biotBC!, fix_resid!, biotBC_r!, pflowBC!, BCTuple
function biot_project!(a::Flow{n},ml_b::MultiLevelPoisson,ω,x₀,tar,ftar,U;fmm=true,w=1,tol=1e-4,itmx=32) where n
    dt = w*a.Δt[end]; a.p .*= dt  # Scale p *= w*Δt
    # dt = w; a.p .*= dt
    apply_grad_p!(a.u,ω,a.p,a.μ₀) # Apply u-=μ₀∇p & ω=∇×u
    x₀ .= a.p; fill!(a.p,0)       # x₀ holds p solution
    biotBC!(a.u,U,ω,tar,ftar;fmm) # Apply domain BCs

    # Set residual
    b = ml_b.levels[1]; b.r .= 0
    @inside b.r[I] = ifelse(b.iD[I]==0,0,WaterLily.div(I,a.u))
    fix_resid!(b.r,a.u,tar[1]) # only fix on the boundaries

    nᵖ,nᵇ,r₂ = 0,0,L₂(b)
    @log ", $nᵖ, $(WaterLily.L∞(b)), $r₂, $nᵇ\n"
    while nᵖ<itmx
        rtol = max(tol,0.1r₂)
        while nᵖ<itmx
            Vcycle!(ml_b); smooth!(b)
            r₂ = L₂(b); nᵖ+=1
            r₂<rtol && break
        end
        apply_grad_p!(a.u,ω,a.p,a.μ₀)   # Update u,ω
        x₀ .+= a.p; fill!(a.p,0)        # Update solution
        biotBC_r!(b.r,a.u,U,ω,tar,ftar;fmm) # Update BC+residual
        r₂ = L₂(b); nᵇ+=1
        @log ", $nᵖ, $(WaterLily.L∞(b)), $r₂, $nᵇ\n"
        r₂<tol && break
    end
    push!(ml_b.n,nᵖ)
    pflowBC!(a.u)  # Update ghost BCs (domain is already correct)
    a.p .= x₀/dt   # copy-scaled pressure solution
end

sim_ref = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)
reset_timer!(to::TimerOutput)

# load aenode struct with trained neural ai models
node_path = "data/NODE_models/Feb12-1551/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)

simdata = load_simdata(aenode.ae_args.full_data_path)


# getting simulation with know initial condition
random_int = 1
u₀, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]
sim_ref.flow.u .= u₀
append!(sim_ref.flow.Δt, simdata.Δt[1:random_int-1])
sim_step!(sim_ref) # perform a sim_step to update all simulation
sim_info(sim_ref)


# flow values before projection
u_first = deepcopy(sim_ref.flow.u)
p_first = deepcopy(sim_ref.flow.p)[2:end-1, 2:end-1]
div_first = mean(Thesis.div_vectorized(sim_ref.flow.u))

# create new simulation for testing projection
project_sim = deepcopy(sim_ref)
project_sim.flow.p .= 0

# doing just biot_project
t₁ = sum(project_sim.flow.Δt)
U = BCTuple(project_sim.flow.uBC,t₁,2); # BCs at t₁
# scale_u!(sim.flow,0.5)
# biot_project!(project_sim.flow,project_sim.pois,project_sim.ω,project_sim.x₀,project_sim.tar,project_sim.ftar, U; fmm=true, w=1)
WaterLily.project!(project_sim.flow, project_sim.pois)


# flow values after projection
u_after = deepcopy(project_sim.flow.u)
p_after = deepcopy(project_sim.flow.p)[2:end-1, 2:end-1]
div_after = mean(Thesis.div_vectorized(project_sim.flow.u))

println("Before projection:")
println("  mean(u) = ", mean(u_first))
println("  mean(p) = ", mean(p_first))
println("  mean(div(u)) = ", div_first, " maximum(div(u)) = $(maximum(Thesis.div_vectorized(project_sim.flow.u)))")

println("\nAfter projection:")
println("  mean(u) = ", mean(u_after))
println("  mean(p) = ", mean(p_after))
println("  mean(div(u)) = ", div_after)


# predicting a single snapshot which should equal the velocity field in sim_ref
u_pred = predict_n(aenode, u₀, μ₀, 1, t₀; Δt=sim_ref.flow.Δt[end])
println("\nMAE between sim and pred: $(mean(abs, u_pred .- sim_ref.flow.u[2:end-1, 2:end-1, :]))\n")


sim_pred = deepcopy(sim_ref)
Thesis.insert_prediction!(sim_pred, u_pred)
push!(sim_pred.flow.Δt, sim_ref.flow.Δt[end])
sim_info(sim_pred)
sim_pred.flow.p .= 0

# predicted simulation fields before projection
u_pred_first = deepcopy(sim_pred.flow.u)
div_pred_first = mean(Thesis.div_vectorized(sim_pred.flow.u))


t₁ = sum(sim_pred.flow.Δt)
U = BCTuple(sim_pred.flow.uBC,t₁,2); # BCs at t₁
# scale_u!(sim.flow,0.5)
# biot_project!(sim_pred.flow,sim_pred.pois,sim_pred.ω,sim_pred.x₀,sim_pred.tar,sim_pred.ftar, U; fmm=true, w=1)
# biot_project!(sim_pred.flow,sim_pred.pois,sim_pred.ω,sim_pred.x₀,sim_pred.tar,sim_pred.ftar, U; fmm=true, w=1)

WaterLily.project!(sim_pred.flow, sim_pred.pois)


# predicted simulation fields after projection
u_pred_after = deepcopy(sim_pred.flow.u)
p_pred_after = deepcopy(sim_pred.flow.p)[2:end-1, 2:end-1]
div_pred_after = mean(Thesis.div_vectorized(sim_pred.flow.u))

println("Before projection:")
println("  mean(u) = ", mean(u_pred_first))
println("  mean(p) = ", mean(p_first))
println("  mean(div(u)) = ", div_pred_first)
# println("  mean(div(u)) = ", div_pred_first, " maximum(div(u)) = $(maximum(Thesis.div_vectorized(sim_pred.flow.u)))")

println("\nAfter projection:")
println("  mean(u) = ", mean(u_pred_after))
println("  mean(p) = ", mean(p_after))
println("  mean(div(u)) = ", div_pred_after, " maximum(div(u)) = $(maximum(Thesis.div_vectorized(sim_pred.flow.u)))")

nothing
