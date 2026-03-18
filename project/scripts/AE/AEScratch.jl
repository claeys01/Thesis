using Thesis

simdata = load_simdata("data/datasets/RE2500/2e8/U_128_full.jld2")
_, normalizer = normalize_batch(simdata.u; normalizer=nothing)
@show normalizer

simdata = load_simdata("data/datasets/RE2500/2e8/U_128_transfer.jld2")
_, normalizer = normalize_batch(simdata.u; normalizer=nothing)
@show normalizer
