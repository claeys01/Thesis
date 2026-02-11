using Thesis
using JLD2

# List of data files to fix
data_files = [
    "data/datasets/RE2500/2e8/U_128_full.jld2",
    # Add other .jld2 files that contain SimData
]

for path in data_files
    @info "Converting $path"
    
    # Load (will auto-convert via updated load_simdata)
    simdata = load_simdata(path)
    
    # Re-save with correct type
    JLD2.save(path, "simdata", simdata)
    
    @info "Saved $path with correct SimData type"
end
