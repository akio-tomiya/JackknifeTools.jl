"""
    JackknifeTools

Small tools for ordinary Jackknife, block Jackknife, histogram error
estimates, and autocorrelation analysis. Use Julia help mode, for example
`?jk_meanerror`, or call `jk_help()` for a short overview.
"""
module JackknifeTools

using Statistics  # Provides functions for statistical calculations

# Functions to export
export jk_index_set, jk_index, jk_meanerror
export jk_block_index_set, jk_block_index, jk_block_meanerror
export jk_replicas, jk_error, jk_cov, jk_meancov
export jk_histogram, jk_block_histogram
export jk_hist, jk_block_hist
export jk_autocorr, jk_tau_int, jk_block_scan
export jk_help

const _HELP_EN = """
JackknifeTools quick help

Ordinary Jackknife:
  jk_meanerror(data)
  jk_meanerror(data, key)
  jk_meanerror(data, func)

Use block Jackknife by setting the block keyword:
  jk_meanerror(data; block=2)
  jk_meanerror(data, key; block=2)
  jk_meanerror(data, func; block=2)

Several correlated series (common Jackknife subsets):
  jk_meanerror((a, b) -> mean(a) / mean(b), A, B; block=2)
  jk_meanerror(A, B; block=2) do a, b
      mean(a) / mean(b)
  end

Vector-valued observables and matrices (rows are samples):
  jk_meanerror(C)                      # column means of C[config, t]
  jk_meancov(f, data; block=2)         # central value and covariance
  rep = jk_replicas(f, data; block=2)  # rep.central, rep.replicas
  jk_error(rep.replicas), jk_cov(rep.replicas)

Histogram with error bars:
  jk_hist(samples; bins=20)
  jk_hist(samples; bins=edges)
  jk_hist(samples; bins=20, block=2)

Autocorrelation:
  jk_autocorr(x, max_lag)
  jk_tau_int(x)                        # automatic windowing
  jk_block_scan(x; blocks=1:50)

Supported keys (String or Symbol):
  "mean", "average"          -> mean(x)
  "sus", "susceptibility"    -> var(x, corrected=false)
  "binder", "bin"            -> Binder ratio, defined here as
                                mean(x .^ 4) / mean(x .^ 2)^2

Histogram results have:
  hist.edges, hist.centers, hist.values, hist.errors
"""

"""
    jk_help([io::IO=stdout]) -> nothing

Prints a short English JackknifeTools help message.
"""
function jk_help(io::IO=stdout)
    print(io, _HELP_EN)
    return nothing
end

include("core.jl")
include("indices.jl")
include("meanerror.jl")
include("histogram.jl")
include("autocorr.jl")

end
