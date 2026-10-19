# Autocorrelation function, integrated autocorrelation time, and block scans.

"""
    jk_autocorr(x::AbstractVector{<:Real}, max_lag::Integer) -> Vector{Float64}

Normalized autocorrelation function `ρ(t) = C(t) / C(0)` for `t = 0:max_lag`,
with

    C(t) = 1 / (N - t) * sum_{i=1}^{N-t} (x_i - x̄) (x_{i+t} - x̄).

The first element is `ρ(0) = 1`. A constant series returns `[1, 0, 0, ...]`.
"""
function jk_autocorr(x::AbstractVector{<:Real}, max_lag::Integer)
    n = length(x)
    n >= 2 || throw(ArgumentError("autocorrelation requires at least two samples"))
    0 <= max_lag <= n - 1 || throw(ArgumentError("max_lag must satisfy 0 <= max_lag <= length(x) - 1"))

    centered = Float64.(x) .- mean(x)
    variance = sum(abs2, centered) / n
    rho = zeros(Float64, max_lag + 1)
    rho[1] = 1.0
    variance == 0 && return rho

    for lag in 1:max_lag
        total = 0.0
        @inbounds for i in 1:n-lag
            total += centered[i] * centered[i+lag]
        end
        rho[lag+1] = total / ((n - lag) * variance)
    end

    return rho
end

"""
    jk_tau_int(x::AbstractVector{<:Real}; window=:auto, c=6, max_lag=length(x) ÷ 2)

Integrated autocorrelation time

    τ_int(W) = 1/2 + sum_{t=1}^{W} ρ(t)

in units of the spacing between consecutive entries of `x`. The variance of the
mean of correlated data is about `2 τ_int Var(x) / N`, so `N / (2 τ_int)` is the
effective number of independent samples.

The summation window `W` is chosen by `window`:
- `:auto` (default): Sokal's automatic windowing, the smallest `W` with
  `W >= c * τ_int(W)`.
- `:first_nonpositive`: stop before the first lag with `ρ(t) <= 0`.
- an integer: use that `W`.

Returns a named tuple `(tau_int, error, window, rho)`. `error` is the
Madras–Sokal estimate `τ_int * sqrt(2 (2W + 1) / N)`, and `rho` is
`jk_autocorr(x, max_lag)`.
"""
function jk_tau_int(x::AbstractVector{<:Real}; window=:auto, c::Real=6, max_lag::Integer=length(x) ÷ 2)
    n = length(x)
    n >= 2 || throw(ArgumentError("tau_int requires at least two samples"))
    max_lag = min(max_lag, n - 1)
    max_lag >= 1 || throw(ArgumentError("max_lag must be at least 1"))
    c > 0 || throw(ArgumentError("c must be positive"))

    rho = jk_autocorr(x, max_lag)
    W = _tau_window(rho, window, c, max_lag)
    tau = 0.5 + sum(@view rho[2:W+1])
    err = tau * sqrt(2 * (2W + 1) / n)

    return (tau_int=tau, error=err, window=W, rho=rho)
end

function _tau_window(rho, window::Integer, c, max_lag)
    window isa Bool && throw(ArgumentError("window must be :auto, :first_nonpositive, or a non-negative integer"))
    0 <= window <= max_lag || throw(ArgumentError("window must satisfy 0 <= window <= max_lag"))
    return Int(window)
end

function _tau_window(rho, window::Symbol, c, max_lag)
    if window === :auto
        tau = 0.5
        for W in 1:max_lag
            tau += rho[W+1]
            W >= c * tau && return W
        end
        return max_lag
    elseif window === :first_nonpositive
        for W in 1:max_lag
            rho[W+1] <= 0 && return W - 1
        end
        return max_lag
    end

    throw(ArgumentError("window must be :auto, :first_nonpositive, or a non-negative integer"))
end

_tau_window(rho, window, c, max_lag) =
    throw(ArgumentError("window must be :auto, :first_nonpositive, or a non-negative integer"))

"""
    jk_block_scan(data; blocks=nothing)
    jk_block_scan(data, func_or_key; blocks=nothing)
    jk_block_scan(func, data1, data2, ...; blocks=nothing)

Block Jackknife estimates for several block sizes. Use it to check that the
error reaches a plateau and that the central value is stable.

`blocks` is a collection of block sizes. The default is `1:max(1, N ÷ 10)`, so
there are at least about 10 blocks. Block sizes that leave fewer than two
blocks are skipped.

Returns a named tuple `(block_sizes, nblocks, values, errors)`. For the mean of
autocorrelated data, `errors[b] / errors[1]` approaches `sqrt(2 τ_int)` once the
block size is much larger than `τ_int`.
"""
function jk_block_scan(func::Function, data1::_Data, data::_Data...; blocks=nothing)
    series = (data1, data...)
    n = _common_nsamples(series)
    sizes = blocks === nothing ? (1:max(1, n ÷ 10)) : blocks

    block_sizes = Int[]
    nblocks = Int[]
    estimates = map(Iterators.filter(b -> div(n, _require_block_size(b; name="blocks")) >= 2, sizes)) do b
        push!(block_sizes, b)
        push!(nblocks, div(n, b))
        _meanerror(func, series, b)
    end
    isempty(block_sizes) && throw(ArgumentError("no block size in blocks leaves at least two blocks"))

    return (
        block_sizes=block_sizes,
        nblocks=nblocks,
        values=first.(estimates),
        errors=last.(estimates),
    )
end

jk_block_scan(data::_Data; blocks=nothing) = jk_block_scan(_default_statistic(data), data; blocks=blocks)
jk_block_scan(data::_Data, func::Function; blocks=nothing) = jk_block_scan(func, data; blocks=blocks)
jk_block_scan(data::_Data, key::_Key; blocks=nothing) = jk_block_scan(_observable_function(key), data; blocks=blocks)
