# Histograms with Jackknife error bars. Each sample contributes one row of bin
# counts; the histogram is the column mean of that count matrix, so the O(N)
# linear-mean path of `_replicas` applies.

function _validate_histogram_edges(edges::AbstractVector{<:Real})
    length(edges) >= 2 || throw(ArgumentError("histogram edges require at least two values"))
    all(isfinite, edges) || throw(ArgumentError("histogram edges must be finite"))

    for i in 1:length(edges)-1
        edges[i] < edges[i+1] || throw(ArgumentError("histogram edges must be strictly increasing"))
    end

    return nothing
end

function _bin_centers(edges::AbstractVector{<:Real})
    return [(edges[i] + edges[i+1]) / 2 for i in 1:length(edges)-1]
end

_sample_values(sample::Real) = (sample,)
_sample_values(sample) = sample

function _histogram_counts!(counts::AbstractVector, values, edges::AbstractVector{<:Real})
    fill!(counts, 0)
    for x in _sample_values(values)
        isnan(x) && continue
        bin = searchsortedlast(edges, x)
        if bin == length(edges) && x == edges[end]
            bin -= 1
        end

        if 1 <= bin <= length(counts)
            counts[bin] += 1
        end
    end

    return counts
end

function _sample_histogram_counts(samples::AbstractVector, edges::AbstractVector{<:Real})
    length(samples) >= 2 || throw(ArgumentError("histogram jackknife requires at least two samples"))
    _validate_histogram_edges(edges)

    counts = zeros(Float64, length(samples), length(edges) - 1)
    for (i, sample) in enumerate(samples)
        _histogram_counts!(view(counts, i, :), sample, edges)
    end

    return counts
end

function _histogram_data_range(samples::AbstractVector)
    xmin = Inf
    xmax = -Inf
    for sample in samples, x in _sample_values(sample)
        isfinite(x) || continue
        xmin = min(xmin, Float64(x))
        xmax = max(xmax, Float64(x))
    end

    xmin <= xmax || throw(ArgumentError("histogram bins cannot be inferred: samples contain no finite values"))
    return xmin, xmax
end

function _histogram_edges_from_bins(samples::AbstractVector, bins::Integer)
    bins isa Bool && throw(ArgumentError("bins must be a positive integer or a vector of edges"))
    bins >= 1 || throw(ArgumentError("bins must be positive"))

    xmin, xmax = _histogram_data_range(samples)
    if xmin == xmax
        half_width = xmin == 0 ? 0.5 : abs(xmin) / 2
        xmin -= half_width
        xmax += half_width
    end

    return collect(range(xmin, xmax; length=bins + 1))
end

function _histogram_edges_from_bins(samples::AbstractVector, bins::AbstractVector{<:Real})
    _validate_histogram_edges(bins)
    return collect(Float64, bins)
end

function _histogram_edges_from_bins(samples::AbstractVector, bins)
    throw(ArgumentError("bins must be a positive integer or a vector of edges"))
end

function _scale_histogram(values::AbstractVector, widths::AbstractVector, scale::Real, density::Bool)
    scaled = Float64.(values) .* Float64(scale)
    density && (scaled ./= widths)
    return scaled
end

function _histogram_from_counts(counts::Matrix{Float64}, edges::AbstractVector{<:Real}, scale::Real, density::Bool, block)
    rep = _replicas(_colmean, (counts,), block)
    widths = Float64[edges[i+1] - edges[i] for i in 1:length(edges)-1]

    # Scaling is linear, so it can be applied to the central value and the error.
    values = _scale_histogram(rep.central, widths, scale, density)
    errors = _scale_histogram(jk_error(rep.replicas), widths, abs(scale), density)

    return (
        edges=Float64.(edges),
        centers=Float64.(_bin_centers(edges)),
        values=values,
        errors=errors,
    )
end

"""
    jk_histogram(samples::AbstractVector, edges::AbstractVector{<:Real}; scale::Real=1.0, density::Bool=false, block=nothing)

Calculates histogram bin values and Jackknife errors.
Set `block` to an integer to use block Jackknife errors.

Each element of `samples` is treated as one Jackknife sample. An element may be a
single real number or a vector of real numbers, such as eigenvalues measured on
one configuration. Bins use `[edge[i], edge[i+1])`, with the final right edge
included. Values outside the edges, `NaN`, and `±Inf` are ignored.

The result is a named tuple with `edges`, `centers`, `values`, and `errors`.
`values` is the average count per sample in each bin. `scale` multiplies both
values and errors. If `density=true`, values and errors are also divided by the
bin widths.

The computation is O(number of samples × number of bins).
"""
function jk_histogram(samples::AbstractVector, edges::AbstractVector{<:Real}; scale::Real=1.0, density::Bool=false, block=nothing)
    counts = _sample_histogram_counts(samples, edges)
    return _histogram_from_counts(counts, edges, scale, density, block)
end

"""
    jk_histogram(samples::AbstractVector; bins=10, scale::Real=1.0, density::Bool=false, block=nothing)

Histogram-like keyword API for `jk_histogram`. `bins` may be a positive integer
or a vector of bin edges. With an integer, the range is taken from the finite
values in `samples`. Set `block` to an integer to use block Jackknife errors.
"""
function jk_histogram(samples::AbstractVector; bins=10, scale::Real=1.0, density::Bool=false, block=nothing)
    edges = _histogram_edges_from_bins(samples, bins)
    return jk_histogram(samples, edges; scale=scale, density=density, block=block)
end

"""
    jk_hist(args...; kwargs...)

Short alias for `jk_histogram`, intended for hist-like usage. For block
Jackknife, use `jk_hist(samples; bins=20, block=2)`.
"""
function jk_hist(args...; kwargs...)
    return jk_histogram(args...; kwargs...)
end

"""
    jk_block_histogram(samples::AbstractVector, edges::AbstractVector{<:Real}, block_size::Integer; scale::Real=1.0, density::Bool=false)

Calculates histogram bin values and block Jackknife errors.
Equivalent to `jk_histogram(samples, edges; block=block_size)`.

If the number of samples is not divisible by `block_size`, the initial remainder
is dropped before making equal-size blocks.
"""
function jk_block_histogram(samples::AbstractVector, edges::AbstractVector{<:Real}, block_size; scale::Real=1.0, density::Bool=false)
    return jk_histogram(samples, edges; scale=scale, density=density, block=block_size)
end

"""
    jk_block_histogram(samples::AbstractVector, block_size::Integer; bins=10, scale::Real=1.0, density::Bool=false)

Histogram-like keyword API for block Jackknife histograms. `bins` may be a
positive integer or a vector of bin edges.
Equivalent to `jk_histogram(samples; bins=bins, block=block_size)`.
"""
function jk_block_histogram(samples::AbstractVector, block_size; bins=10, scale::Real=1.0, density::Bool=false)
    return jk_histogram(samples; bins=bins, scale=scale, density=density, block=block_size)
end

"""
    jk_block_hist(args...; kwargs...)

Short alias for `jk_block_histogram`, intended for hist-like usage. The
preferred unified form is `jk_hist(samples; bins=20, block=2)`.
"""
function jk_block_hist(args...; kwargs...)
    return jk_block_histogram(args...; kwargs...)
end
