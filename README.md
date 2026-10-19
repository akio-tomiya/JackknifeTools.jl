<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/logo-dark.svg">
    <img alt="JackknifeTools" src="docs/src/assets/logo.svg" width="520">
  </picture>
</p>

[![CI](https://github.com/akio-tomiya/JackknifeTools.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/akio-tomiya/JackknifeTools.jl/actions/workflows/CI.yml)

**Author**: A. Tomiya

[Japanese manual](README.ja.md)

JackknifeTools is a small Julia package for Jackknife and block Jackknife error
estimates of Monte Carlo data. It gives central values and errors for primary
observables, for nonlinear secondary observables, for observables built from
several correlated series, and for histograms. It also estimates
autocorrelation times and scans block sizes.

The package depends only on Julia's standard `Statistics` library. Memory use
is O(N) in the number of samples, and the mean and histograms take O(N) time,
so long Monte Carlo histories are fine.

The package was called JkTools before version 1.0. The function names
(`jk_meanerror`, `jk_hist`, ...) are unchanged.

## Installation

```julia
pkg> add JackknifeTools
```

Until the registration in the General registry is complete, install from
GitHub:

```julia
pkg> add https://github.com/akio-tomiya/JackknifeTools.jl
```

## Quick start

```julia
using JackknifeTools
using Statistics

data = randn(10_000)
A = 3 .+ data .+ 0.1 .* randn(10_000)   # two series measured on the same samples
B = 2 .+ data .+ 0.1 .* randn(10_000)

jk_meanerror(data)                                  # mean and Jackknife error
jk_meanerror(data; block=50)                        # block Jackknife
jk_meanerror(data, x -> var(x, corrected=false))    # nonlinear observable
jk_meanerror(data, "binder"; block=50)              # predefined key
jk_meanerror(A, B; block=50) do a, b                # several correlated series
    mean(a) / mean(b)
end
```

Every estimator returns `(central_value, error)`. The central value is the
statistic of the full sample; the error comes from the Jackknife replicas.

## Features

| Task | Functions |
| --- | --- |
| Mean and error | `jk_meanerror(data)`, `jk_meanerror(data, func)`, `jk_meanerror(data, key)` |
| Several correlated series | `jk_meanerror(func, A, B, ...)` and the do-block form |
| Vector observables, covariance | `jk_meanerror(data, func)` with array-valued `func`, `jk_meancov` |
| Jackknife replicas | `jk_replicas`, `jk_error`, `jk_cov` |
| Histograms with error bars | `jk_hist(samples; bins=20)`, `jk_histogram(samples, edges)` |
| Autocorrelation | `jk_autocorr`, `jk_tau_int`, `jk_block_scan` |
| Index sets | `jk_index`, `jk_index_set`, `jk_block_index`, `jk_block_index_set` |
| Older explicit block API | `jk_block_meanerror`, `jk_block_histogram`, `jk_block_hist` |
| Help | `?jk_meanerror`, `?jk_hist`, `jk_help()` |

Every estimator accepts the `block` keyword. Without it, ordinary
leave-one-out Jackknife is used.

Supported observable keys (`String` or `Symbol`):

```julia
"mean", "average"          # mean(x)
"sus", "susceptibility"    # var(x, corrected=false)
"binder", "bin"            # Binder ratio: mean(x .^ 4) / mean(x .^ 2)^2
```

## Data layout

- A vector has one entry per sample (configuration).
- A matrix has one row per sample, for example a correlator `C[config, t]`.
  `jk_meanerror(C)` gives the column means and their errors.
- A vector of vectors also works; each inner vector belongs to one sample.
- Complex data give the error `complex(error of real part, error of imaginary part)`.

`func` receives the samples in the same layout and may return a number, a
complex number, or an array. Arrays give elementwise errors.

## Block Jackknife

Block Jackknife removes one contiguous block at a time. Use it when the data
are autocorrelated, as in Markov-chain Monte Carlo.

```julia
jk_meanerror(data; block=50)
jk_meanerror(data, "sus"; block=50)
```

If `length(data)` is not divisible by the block size, JackknifeTools drops the
initial remainder before making equal-size blocks. The central value and the
replicas both use the data after the drop.

```julia
jk_block_index_set(1:5, 2)
# [[4, 5], [2, 3]]
```

Here `1` is dropped, and the kept data are split into `[2, 3]` and `[4, 5]`.
At least two full blocks are required.

### Choosing the block size

```julia
tau = jk_tau_int(data)          # automatic windowing (Sokal), c = 6
tau.tau_int, tau.error, tau.window

scan = jk_block_scan(data; blocks=1:200)
scan.block_sizes, scan.nblocks, scan.values, scan.errors
```

The block Jackknife error of a mean grows with the block size and levels off
near `sqrt(2 tau_int)` times the ordinary error once the block size is much
larger than `tau_int`. Pick a block size in that plateau that still leaves
many blocks.

## Several correlated series

When an observable is built from several quantities measured on the same
configurations, pass all series to one call. Every replica removes the same
samples (or the same block) from all series, so their correlations enter the
error.

```julia
ratio, ratio_err = jk_meanerror((a, b) -> mean(a) / mean(b), A, B; block=50)
```

Estimating the errors of `mean(A)` and `mean(B)` separately and combining them
as independent can be wrong by a large factor. The tutorial shows a case where
it overestimates the error by more than a factor of 10.

## Vector observables, covariance, and replicas

```julia
# effective mass from a correlator C[config, t]
meff(c) = (ct = vec(mean(c; dims=1)); log.(ct[1:end-1] ./ ct[2:end]))

m, m_err = jk_meanerror(C, meff; block=4)
m, m_cov = jk_meancov(C, meff; block=4)        # covariance for correlated fits

rep = jk_replicas(C, meff; block=4)            # one value per Jackknife sample
jk_error(rep.replicas), jk_cov(rep.replicas)
```

## Histograms with error bars

Each element of `samples` is one sample. It may be a number or a vector, for
example the Dirac eigenvalues measured on one gauge configuration.

```julia
eigenvalues_by_config = [
    [0.002, 0.011, 0.018],
    [0.004, 0.010, 0.026],
    [0.001, 0.016, 0.021],
]

hist = jk_hist(eigenvalues_by_config; bins=collect(0.0:0.01:0.03))
hist.edges, hist.centers, hist.values, hist.errors
```

`bins` may be an integer or a vector of edges. Bins are `[edge[i], edge[i+1])`
with the last right edge included. Values outside the edges, `NaN`, and
`±Inf` are ignored. Use `density=true` to divide by the bin width and `scale`
for any other normalization:

```julia
volume = 32^3 * 8
hist = jk_hist(eigenvalues_by_config; bins=20, block=2, scale=1 / volume, density=true)
```

JackknifeTools does not depend on a plotting package. With Plots.jl:

```julia
using Plots
bar(hist.centers, hist.values; yerror=hist.errors, bar_width=hist.edges[2] - hist.edges[1], label=false)
```

See `examples/histogram_errorbars.jl`.

## Tutorial

[`tutorials/README.md`](tutorials/README.md) is a Japanese tutorial on the 2D
Ising model with a heat-bath update. It covers statistical errors, error
propagation, ordinary and block Jackknife, autocorrelation functions and
times, the choice of the block size, observables from several correlated
series, and histograms, with saved output and figures.

```bash
julia --project=. tutorials/ising2d_jackknife.jl
julia --project=. tutorials/make_tutorial_artifacts.jl
```

## Input validation

Ordinary Jackknife requires at least two samples. Block Jackknife requires a
positive integer block size and at least two full blocks after dropping the
initial remainder. Several series must have the same number of samples.
Invalid inputs throw `ArgumentError`.

## License

MIT License.
