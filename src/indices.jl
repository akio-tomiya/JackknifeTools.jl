# Explicit Jackknife index subsets. The estimators do not use these functions,
# because storing every subset needs O(N^2) memory. They are kept for users who
# want to resample by hand.

"""
    jk_index(input_data::AbstractVector; block=nothing) -> Vector{Vector{Int}}

Generates Jackknife index subsets directly from the input data.
Set `block` to an integer to generate block Jackknife index subsets.

Returns:
`Vector{Vector{Int}}`. Each element is an index set with one data point removed.
All subsets are stored, so memory grows as `length(input_data)^2`. The
estimators such as `jk_meanerror` do not call this function.

Example:
```julia
jk_index([10.0, 20.0, 30.0])
# [[2, 3], [1, 3], [1, 2]]

jk_index([10.0, 20.0, 30.0, 40.0]; block=2)
# [[3, 4], [1, 2]]
```
"""
function jk_index(input_data::AbstractVector; block=nothing)
    Ndat = length(input_data)
    index = 1:Ndat
    block_size = _optional_block_size(block)
    return block_size === nothing ? jk_index_set(index) : jk_block_index_set(index, block_size)
end

"""
    jk_index_set(index::AbstractVector{<:Integer}; block=nothing) -> Vector{Vector{Int}}

Generates subsets of indices for Jackknife resampling from the given array of indices.
Set `block` to an integer to generate block Jackknife index subsets.

Example:
```julia
jk_index_set(1:4)
# [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]]

jk_index_set(1:5; block=2)
# [[4, 5], [2, 3]]
```
"""
function jk_index_set(index::AbstractVector{<:Integer}; block=nothing)
    block_size = _optional_block_size(block)
    block_size === nothing || return jk_block_index_set(index, block_size)

    N = length(index)  # Total number of indices
    subsets = Vector{Vector{Int}}()  # Container for subsets
    sizehint!(subsets, N)

    # Create subsets by excluding one index at a time
    for i in 1:N
        subset = Vector{Int}(vcat(index[1:i-1], index[i+1:end]))
        push!(subsets, subset)
    end
    return subsets
end

"""
    jk_block_index(input_data::AbstractVector, block_size::Integer) -> Vector{Vector{Int}}

Generates block Jackknife index subsets directly from the input data.

If the input length is not divisible by `block_size`, the initial remainder is
dropped before making equal-size blocks.
"""
function jk_block_index(input_data::AbstractVector, block_size)
    return jk_index(input_data; block=block_size)
end

"""
    jk_block_index_set(index::AbstractVector{<:Integer}, block_size::Integer) -> Vector{Vector{Int}}

Generates leave-one-block-out index subsets for block Jackknife resampling.

If `length(index)` is not divisible by `block_size`, the initial remainder is
dropped before making equal-size blocks.
"""
function jk_block_index_set(index::AbstractVector{<:Integer}, block_size)
    N = length(index)
    first_kept, nblocks = _block_first_kept(N, block_size)

    subsets = Vector{Vector{Int}}()
    sizehint!(subsets, nblocks)

    for block_start in first_kept:block_size:N
        block_end = block_start + block_size - 1
        subset = Vector{Int}(vcat(index[first_kept:block_start-1], index[block_end+1:N]))
        push!(subsets, subset)
    end

    return subsets
end
