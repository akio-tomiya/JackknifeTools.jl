# Estimators: jk_replicas, jk_meanerror, jk_meancov, and the older explicit
# block API.

const _Key = Union{AbstractString,Symbol}

function _observable_function(key::_Key)
    KEY = String(key)
    return if KEY in ("mean", "average")
        mean
    elseif KEY in ("sus", "susceptibility")
        x -> var(x, corrected=false)
    elseif KEY in ("binder", "bin")
        x -> mean(x .^ 4) / mean(x .^ 2)^2
    else
        throw(ArgumentError("unsupported observable key: $KEY. Supported keys are \"mean\", \"average\", \"sus\", \"susceptibility\", \"binder\", and \"bin\""))
    end
end

"""
    jk_replicas(data; block=nothing) -> (central, replicas)
    jk_replicas(data, func_or_key; block=nothing) -> (central, replicas)
    jk_replicas(func, data1, data2, ...; block=nothing) -> (central, replicas)

Returns the central value and the Jackknife replicas `θ_i` as a named tuple.
Pass the replicas to `jk_error` or `jk_cov`, or to a later fit that needs one
value per Jackknife sample.

`data` is a vector with one entry per sample (configuration), or a matrix whose
rows are samples, such as a correlator `C[config, t]`. For a matrix the default
statistic is the column mean.

With several data series, `func` receives one argument per series, and every
replica removes the same samples (or the same block) from all series, so the
correlations between the series are kept. Do-block syntax works:

```julia
rep = jk_replicas(A, B; block=10) do a, b
    mean(a) / mean(b)
end
rep.central, jk_error(rep.replicas)
```

The central value is `func` applied to all kept samples, not the mean of the
replicas. With `block`, the initial `length(data) % block` samples are dropped
before making blocks, for both the central value and the replicas.
"""
function jk_replicas(statistic::Function, data1::_Data, data::_Data...; block=nothing)
    return _replicas(statistic, (data1, data...), block)
end

jk_replicas(data::_Data; block=nothing) = jk_replicas(_default_statistic(data), data; block=block)
jk_replicas(data::_Data, func::Function; block=nothing) = jk_replicas(func, data; block=block)
jk_replicas(data::_Data, key::_Key; block=nothing) = jk_replicas(_observable_function(key), data; block=block)

function _meanerror(statistic, data::Tuple, block)
    rep = _replicas(statistic, data, block)
    return rep.central, jk_error(rep.replicas)
end

"""
    jk_meanerror(input_data; block=nothing) -> (central_value, error)

Calculates the mean and Jackknife error for the given data.
Set `block` to an integer to use block Jackknife.

`input_data` is usually a real vector with one value per sample. It may also be
a complex vector, a vector of vectors, or a matrix whose rows are samples. For a
matrix, the result is the column mean and its elementwise error.

The mean uses an O(N) formula, so long Monte Carlo histories are fine.

Returns:
`(central_value, error)`.
"""
function jk_meanerror(input_data::_Data; block=nothing)
    return _meanerror(_default_statistic(input_data), (input_data,), block)
end

"""
    jk_meanerror(input_data, func::Function; block=nothing) -> (central_value, error)

Calculates `func(input_data)` and its Jackknife error.
`func` maps one sample vector (or a matrix of sample rows) to a scalar, a
complex number, or an array. Arrays give an elementwise error.
Set `block` to an integer to use block Jackknife.

Example:
```julia
jk_meanerror(data, x -> mean(x .^ 2))
jk_meanerror(data, x -> mean(x .^ 2); block=2)
jk_meanerror(data, x -> var(x, corrected=false))
jk_meanerror(data, x -> [mean(x), mean(x .^ 2)])  # vector-valued
```

Memory use is O(N): one leave-out buffer is reused for all Jackknife samples.
The time is O(N) calls of `func`.
"""
function jk_meanerror(input_data::_Data, func::Function; block=nothing)
    return _meanerror(func, (input_data,), block)
end

"""
    jk_meanerror(input_data, KEY; block=nothing) -> (central_value, error)

Calculates the mean and Jackknife error for statistical observables based on a specified keyword.
`KEY` may be a `String`, another `AbstractString`, or a `Symbol`.
Set `block` to an integer to use block Jackknife.

Supported keys:
- `"mean"` or `"average"`: `mean(x)`
- `"sus"` or `"susceptibility"`: `var(x, corrected=false)`
- `"binder"` or `"bin"`: Binder ratio, defined in JackknifeTools as
  `mean(x .^ 4) / mean(x .^ 2)^2`

Unsupported keys throw an `ArgumentError`.
"""
function jk_meanerror(input_data::_Data, KEY::_Key; block=nothing)
    return _meanerror(_observable_function(KEY), (input_data,), block)
end

"""
    jk_meanerror(func::Function, data1, data2, ...; block=nothing) -> (central_value, error)

Jackknife estimate of an observable built from several correlated series, for
example `mean(A) / mean(B)` where `A` and `B` were measured on the same
configurations. Every Jackknife sample removes the same samples (or the same
block) from all series, so their correlations enter the error.

```julia
ratio, ratio_err = jk_meanerror((a, b) -> mean(a) / mean(b), A, B; block=10)

ratio, ratio_err = jk_meanerror(A, B; block=10) do a, b
    mean(a) / mean(b)
end
```

All series must have the same number of samples. Matrices are allowed; their
rows are samples.
"""
function jk_meanerror(func::Function, data1::_Data, data::_Data...; block=nothing)
    return _meanerror(func, (data1, data...), block)
end

"""
    jk_meancov(data; block=nothing) -> (central_value, covariance)
    jk_meancov(data, func_or_key; block=nothing) -> (central_value, covariance)
    jk_meancov(func, data1, data2, ...; block=nothing) -> (central_value, covariance)

Like `jk_meanerror`, but returns the Jackknife covariance matrix of a real,
possibly vector-valued, observable. The central value is returned as computed;
the covariance has size `length(central) × length(central)`.

Example: effective mass from a correlator `C[config, t]`,
```julia
meff(c) = (ct = vec(mean(c; dims=1)); log.(ct[1:end-1] ./ ct[2:end]))
m, cov = jk_meancov(C, meff; block=4)
```
"""
function jk_meancov(func::Function, data1::_Data, data::_Data...; block=nothing)
    rep = _replicas(func, (data1, data...), block)
    return rep.central, jk_cov(rep.replicas)
end

jk_meancov(data::_Data; block=nothing) = jk_meancov(_default_statistic(data), data; block=block)
jk_meancov(data::_Data, func::Function; block=nothing) = jk_meancov(func, data; block=block)
jk_meancov(data::_Data, key::_Key; block=nothing) = jk_meancov(_observable_function(key), data; block=block)

"""
    jk_block_meanerror(input_data, block_size::Integer) -> (central_value, error)

Calculates the mean and block Jackknife error for the given data array.
Equivalent to `jk_meanerror(input_data; block=block_size)`.
"""
function jk_block_meanerror(input_data::_Data, block_size)
    return jk_meanerror(input_data; block=block_size)
end

"""
    jk_block_meanerror(input_data, block_size::Integer, func::Function) -> (central_value, error)

Calculates a custom statistic and its block Jackknife error.
Equivalent to `jk_meanerror(input_data, func; block=block_size)`.
"""
function jk_block_meanerror(input_data::_Data, block_size, func::Function)
    return jk_meanerror(input_data, func; block=block_size)
end

"""
    jk_block_meanerror(input_data, block_size::Integer, KEY) -> (central_value, error)

Calculates a predefined observable and its block Jackknife error.
Equivalent to `jk_meanerror(input_data, KEY; block=block_size)`.

Uses the same keys as `jk_meanerror`. `"binder"` and `"bin"` mean the Binder
ratio, defined in JackknifeTools as `mean(x .^ 4) / mean(x .^ 2)^2`.
"""
function jk_block_meanerror(input_data::_Data, block_size, KEY::_Key)
    return jk_meanerror(input_data, KEY; block=block_size)
end
