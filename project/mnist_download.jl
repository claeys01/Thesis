using Plots
gr()  # or pyplot(), if you prefer Matplotlib style

# Fake data (replace with your real values)
epochs = 1:60
loss = @. 1e-2 * exp(-0.2 * epochs) + 1e-6
cc1  = @. 0.9 * (1 - exp(-0.3 * epochs))
cc2  = @. 0.85 * (1 - exp(-0.25 * epochs))

# Plot loss (left axis, black line)
p1 = plot(epochs, loss;
    yaxis = :log,
    color = :black,
    lw = 1.5,
    xlabel = "Epoch",
    ylabel = "Loss",
    legend = false)

# Add CC curves on the right axis
p2 = twinx()  # create secondary y-axis
plot!(p2, epochs, cc1;
    color = :blue,
    lw = 1.5,
    ylabel = "CC",
    ylim = (0,1),
    legend = false)
plot!(p2, epochs, cc2;
    color = :purple,
    lw = 1.0)

display(p1)
