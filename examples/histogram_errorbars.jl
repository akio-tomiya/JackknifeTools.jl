using JackknifeTools
using Plots
using Random

Random.seed!(20260515)

# Example data layout:
# one element of `eigenvalues_by_config` corresponds to one configuration.
eigenvalues_by_config = [
    sort!(0.002 .+ 0.04 .* rand(12) .+ 0.0005 * cfg)
    for cfg in 1:24
]

block_size = 4
bins = collect(0.0:0.005:0.06)

hist = jk_hist(eigenvalues_by_config; bins=bins, block=block_size, density=true)

bar(
    hist.centers,
    hist.values;
    yerror=hist.errors,
    bar_width=hist.edges[2] - hist.edges[1],
    label=false,
    xlabel="lambda",
    ylabel="rho(lambda)",
    title="Block-jackknife histogram",
)
