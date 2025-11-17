using Lux
# using Lux.GraphUtils
using GraphViz

model = Chain(Dense(4 => 16, relu), Dense(16 => 2))

# Produce GraphViz graph
g = computational_graph(model)

# Save as PNG
GraphViz.output(PNG("model.png"), g)
