using Thesis

args = LuxArgs()
Thesis.get_data(
    args.batch_size,
    args.full_data_path;
    n_training = args.train_downsample,
    n_test = args.test_downsample,
    split = args.split,
    t_training = args.t_training,
)

# simdata = load_simdata("data/datasets/RE2500/2e8/U_128_full.jld2")
# _, normalizer = normalize_batch(simdata.u; normalizer=nothing)
# @show normalizer

# simdata = load_simdata("data/datasets/RE2500/2e8/U_128_transfer.jld2")
# _, normalizer = normalize_batch(simdata.u; normalizer=nothing)
# @show normalizer
nothing
