# Core Jackknife machinery shared by the scalar, vector, matrix, and histogram
# APIs. Ordinary Jackknife is treated as block Jackknife with block size 1.

const _Data = Union{AbstractVector,AbstractMatrix}

function _require_block_size(block_size; name::String="block_size")
    block_size isa Integer && !(block_size isa Bool) || throw(ArgumentError("$name must be a positive integer"))
    block_size >= 1 || throw(ArgumentError("$name must be positive"))
    return block_size
end

function _optional_block_size(block)
    block === nothing && return nothing
    return _require_block_size(block; name="block")
end

_nsamples(x::AbstractVector) = length(x)
_nsamples(x::AbstractMatrix) = size(x, 1)

_samples(x::AbstractVector, range) = x[range]
_samples(x::AbstractMatrix, range) = x[range, :]

function _block_first_kept(n::Integer, block_size)
    block_size = _require_block_size(block_size)

    remainder = mod(n, block_size)
    first_kept = remainder + 1
    nblocks = div(n - remainder, block_size)
    nblocks >= 2 || throw(ArgumentError("block jackknife requires at least two blocks"))

    return first_kept, nblocks
end

# Returns (first_kept, nblocks, block_size). The data used for the central value
# and for every Jackknife sample are data[first_kept:n].
function _layout(n::Integer, block)
    block_size = _optional_block_size(block)
    if block_size === nothing
        n >= 2 || throw(ArgumentError("jackknife error requires at least two data points"))
        return 1, n, 1
    end

    first_kept, nblocks = _block_first_kept(n, block_size)
    return first_kept, nblocks, block_size
end

function _common_nsamples(data::Tuple)
    for x in data
        Base.require_one_based_indexing(x)
    end

    n = _nsamples(first(data))
    for x in data
        _nsamples(x) == n || throw(ArgumentError("all data series must have the same number of samples"))
    end

    return n
end

_leaveout_buffer(x::AbstractVector, m::Integer) = similar(x, m)
_leaveout_buffer(x::AbstractMatrix, m::Integer) = similar(x, m, size(x, 2))

# Fill `buf` with x[first_kept:n] after removing the block start:start+b-1.
function _fill_leaveout!(buf::AbstractVector, x::AbstractVector, first_kept, start, b, n)
    nbefore = start - first_kept
    nbefore > 0 && copyto!(buf, 1, x, first_kept, nbefore)
    nafter = n - (start + b) + 1
    nafter > 0 && copyto!(buf, nbefore + 1, x, start + b, nafter)
    return buf
end

function _fill_leaveout!(buf::AbstractMatrix, x::AbstractMatrix, first_kept, start, b, n)
    nbefore = start - first_kept
    @views buf[1:nbefore, :] .= x[first_kept:start-1, :]
    @views buf[nbefore+1:end, :] .= x[start+b:n, :]
    return buf
end

# Leave-one-out buffers are reused, so results that are arrays are copied in
# case they alias the buffer, for example `x -> view(x, 1:3)`.
_detach(value::AbstractArray) = copy(value)
_detach(value) = value

_colmean(x::AbstractMatrix) = vec(mean(x; dims=1))

_default_statistic(::AbstractVector) = mean
_default_statistic(::AbstractMatrix) = _colmean

# Linear statistics have O(N) Jackknife samples built from block sums.
_is_linear_mean(statistic, data::Tuple{AbstractVector{<:Number}}) = statistic === mean
_is_linear_mean(statistic, data::Tuple{AbstractMatrix{<:Number}}) = statistic === _colmean
_is_linear_mean(statistic, data) = false

function _linear_mean_replicas(x::AbstractVector, first_kept, nblocks, b, n)
    block_sums = [sum(view(x, start:start+b-1)) for start in first_kept:b:n]
    total = sum(block_sums)
    nleft = (nblocks - 1) * b
    central = first_kept == 1 ? mean(x) : mean(view(x, first_kept:n))
    return central, [(total - block_sum) / nleft for block_sum in block_sums]
end

function _linear_mean_replicas(x::AbstractMatrix, first_kept, nblocks, b, n)
    block_sums = [vec(sum(view(x, start:start+b-1, :); dims=1)) for start in first_kept:b:n]
    total = sum(block_sums)
    nleft = (nblocks - 1) * b
    central = _colmean(view(x, first_kept:n, :))
    return central, [(total .- block_sum) ./ nleft for block_sum in block_sums]
end

function _replicas(statistic, data::Tuple, block)
    n = _common_nsamples(data)
    first_kept, nblocks, b = _layout(n, block)

    if _is_linear_mean(statistic, data)
        central, replicas = _linear_mean_replicas(only(data), first_kept, nblocks, b, n)
        return (central=central, replicas=replicas)
    end

    kept = first_kept == 1 ? data : map(x -> _samples(x, first_kept:n), data)
    central = statistic(kept...)

    buffers = map(x -> _leaveout_buffer(x, n - first_kept + 1 - b), data)
    replicas = map(first_kept:b:n) do start
        for (buf, x) in zip(buffers, data)
            _fill_leaveout!(buf, x, first_kept, start, b, n)
        end
        _detach(statistic(buffers...))
    end

    return (central=central, replicas=replicas)
end

# Squares and square roots taken separately for real and imaginary parts, so a
# complex observable gets the error complex(err_real, err_imag).
_sqparts(x::Real) = abs2(x)
_sqparts(z::Complex) = complex(abs2(real(z)), abs2(imag(z)))
_sqrtparts(x::Real) = sqrt(x)
_sqrtparts(z::Complex) = complex(sqrt(real(z)), sqrt(imag(z)))

function _check_replicas(replicas::AbstractVector)
    length(replicas) >= 2 || throw(ArgumentError("at least two Jackknife replicas are required"))
    return length(replicas)
end

"""
    jk_error(replicas::AbstractVector) -> error

Jackknife error computed from Jackknife replicas `θ_i`:

    sqrt((n - 1) / n * sum_i (θ_i - θ̄)^2),   θ̄ = mean(θ_i).

Replicas may be real numbers, complex numbers, or arrays of the same shape.
Arrays give an elementwise error. Complex values give
`complex(error of real part, error of imaginary part)`.

Example:
```julia
rep = jk_replicas(data; block=4)
jk_error(rep.replicas) == jk_meanerror(data; block=4)[2]
```
"""
function jk_error(replicas::AbstractVector)
    n = _check_replicas(replicas)
    center = mean(replicas)
    sumsq = sum(r -> _sqparts.(r .- center), replicas)
    return _sqrtparts.((n - 1) / n .* sumsq)
end

"""
    jk_cov(replicas::AbstractVector) -> Matrix

Jackknife covariance matrix computed from real-valued Jackknife replicas:

    (n - 1) / n * sum_i (θ_i - θ̄) (θ_i - θ̄)^T.

Each replica may be a real number or a real array. Arrays are flattened with
`vec`, so the result has size `length(θ) × length(θ)`. The diagonal equals
`jk_error(replicas) .^ 2`. Use this matrix for correlated fits.
"""
function jk_cov(replicas::AbstractVector)
    n = _check_replicas(replicas)
    vectors = [_real_vector(r) for r in replicas]
    dim = length(first(vectors))
    all(v -> length(v) == dim, vectors) || throw(ArgumentError("all replicas must have the same length"))

    center = mean(vectors)
    cov = zeros(Float64, dim, dim)
    for v in vectors
        d = v .- center
        cov .+= d * d'
    end

    return (n - 1) / n .* cov
end

_real_vector(x::Real) = [Float64(x)]
_real_vector(x::AbstractArray{<:Real}) = Float64.(vec(x))
_real_vector(x) = throw(ArgumentError("jk_cov requires real-valued replicas; split complex observables into real and imaginary parts"))
