
using MLDatasets: MNIST

dataset = MNIST(; split = :train)[1:2000] # Partial load for demonstration
println(size(dataset[1]))