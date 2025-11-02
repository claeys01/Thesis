ENV["JULIA_CUDA_USE_CUDNN"] = "false"


using Flux
using CUDA
# using cuDNN    # make sure cuDNN is actually loaded so convs use it

# 1. Basic GPU sanity checks
println("CUDA.has_cuda()      = ", CUDA.has_cuda())
println("CUDA.functional()    = ", CUDA.functional())
println("CUDA toolkit version = ", CUDA.versioninfo())

# If this prints false, stop here. No point continuing.
@assert CUDA.has_cuda() "No CUDA device detected"
@assert CUDA.functional() "CUDA not functional (driver / toolkit mismatch?)"

CUDA.allowscalar(false)  # catch slow accidental CPU fallbacks

# 2. Define a tiny ConvNet
# Input shape we'll use: (28, 28, 1, batch)
model = Chain(
    Conv((3,3), 1 => 8; pad=1),      # 28x28x1 -> 28x28x8
    relu,
    MaxPool((2,2)),                  # 28x28x8 -> 14x14x8
    Conv((3,3), 8 => 16; pad=1),     # 14x14x8 -> 14x14x16
    relu,
    MaxPool((2,2)),                  # 14x14x16 -> 7x7x16
    Flux.flatten,                         # 7*7*16 = 784
    Dense(7*7*16, 10),               # 10 classes
    softmax
)

println("Model (CPU):")
println(model)

# 3. Move model and data to GPU
m_gpu = model |> gpu

# Fake input batch: 32 grayscale "images" of size 28x28
x_cpu = rand(Float32, 28,28,1,32)
# Fake labels: integers 1..10
y_idx = rand(1:10, 32)

# We'll train with logit cross entropy, so make one-hot targets
y_cpu = Flux.onehotbatch(y_idx, 1:10)

# ship to GPU
x_gpu = cu(x_cpu)
y_gpu = cu(y_cpu)

println("x_gpu type: ", typeof(x_gpu))
println("y_gpu type: ", typeof(y_gpu))
println("First param type: ", typeof(first(Flux.trainable(m_gpu))))

# 4. Define loss
loss(m, x, y) = Flux.Losses.logitcrossentropy(m(x), y)

# 5. Forward pass test
ℓ = loss(m_gpu, x_gpu, y_gpu)
println("Forward loss on GPU = ", ℓ)

# 6. One training step on GPU
opt = Flux.setup(Flux.Descent(1e-3), m_gpu)

grads = Flux.gradient(() -> loss(m_gpu, x_gpu, y_gpu), Flux.params(m_gpu))
Flux.update!(opt, Flux.params(m_gpu), grads)

println("Did one backward + update on GPU ✅")

# 7. Extra confirmation: run again after update
ℓ2 = loss(m_gpu, x_gpu, y_gpu)
println("Loss after one step = ", ℓ2)

println("Success: Flux + CUDA + cuDNN all ran on GPU without throwing.")
