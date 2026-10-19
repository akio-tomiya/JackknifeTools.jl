using JackknifeTools
using Statistics
using Test
using Random
using Aqua

include(joinpath(@__DIR__, "..", "tutorials", "ising2d_jackknife.jl"))

function normalize_doc_whitespace(text::AbstractString)
    return replace(text, r"\s+" => " ")
end

function jackknife_expected(data::AbstractVector, statistic::Function)
    n = length(data)
    n >= 2 || throw(ArgumentError("jackknife error requires at least two data points"))

    jk_values = [
        statistic(data[vcat(1:i-1, i+1:n)])
        for i in 1:n
    ]

    return statistic(data), std(jk_values, corrected=false) * sqrt(n - 1)
end

function block_jackknife_expected(data::AbstractVector, block_size::Integer, statistic::Function)
    n = length(data)
    block_size >= 1 || throw(ArgumentError("block_size must be positive"))

    first_kept = mod(n, block_size) + 1
    block_ranges = [
        start:start + block_size - 1
        for start in first_kept:block_size:n
    ]
    length(block_ranges) >= 2 || throw(ArgumentError("block jackknife requires at least two blocks"))

    jk_values = [
        statistic(data[vcat(first_kept:first(block)-1, last(block)+1:n)])
        for block in block_ranges
    ]

    return statistic(data[first_kept:n]), std(jk_values, corrected=false) * sqrt(length(block_ranges) - 1)
end

function test_result(got, expected)
    @test got[1] ≈ expected[1]
    @test got[2] ≈ expected[2]
end

@testset "JackknifeTools.jl" begin
    @testset "jackknife indices" begin
        @test jk_index_set([1, 2, 3, 4]) == [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]]
        @test jk_index_set(1:4) == [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]]
        @test jk_index_set([2, 4, 6]) == [[4, 6], [2, 6], [2, 4]]

        @test jk_index([10.0, 20.0, 30.0]) == [[2, 3], [1, 3], [1, 2]]
        @test jk_index([10, 20, 30]) == [[2, 3], [1, 3], [1, 2]]
        @test jk_index_set(1:5; block=2) == jk_block_index_set(1:5, 2)
        @test jk_index([10.0, 20.0, 30.0, 40.0, 50.0]; block=2) == jk_block_index([10.0, 20.0, 30.0, 40.0, 50.0], 2)
    end

    @testset "mean jackknife error" begin
        data = [1.0, 2.0, 3.0, 4.0]
        expected = jackknife_expected(data, mean)

        test_result(jk_meanerror(data), expected)
        test_result(jk_meanerror(data, "mean"), expected)
        test_result(jk_meanerror(data, "average"), expected)
    end

    @testset "observable keys" begin
        data = [1.0, 2.0, 3.0, 4.0]

        susceptibility = x -> var(x, corrected=false)
        susceptibility_expected = jackknife_expected(data, susceptibility)
        test_result(jk_meanerror(data, "sus"), susceptibility_expected)
        test_result(jk_meanerror(data, "susceptibility"), susceptibility_expected)

        binder_ratio = x -> mean(x .^ 4) / mean(x .^ 2)^2
        binder_expected = jackknife_expected(data, binder_ratio)
        test_result(jk_meanerror(data, "binder"), binder_expected)
        test_result(jk_meanerror(data, "bin"), binder_expected)

        @test_throws ArgumentError jk_meanerror(data, "unknown")
    end

    @testset "custom statistic" begin
        data = [1.0, 2.0, 3.0, 4.0]

        square_mean = x -> mean(x .^ 2)
        test_result(jk_meanerror(data, square_mean), jackknife_expected(data, square_mean))

        susceptibility = x -> var(x, corrected=false)
        test_result(jk_meanerror(data, susceptibility), jackknife_expected(data, susceptibility))
    end

    @testset "real-valued inputs" begin
        data = [1, 2, 3, 4]
        test_result(jk_meanerror(data), jackknife_expected(data, mean))
        test_result(jk_meanerror(data, "sus"), jackknife_expected(data, x -> var(x, corrected=false)))
    end

    @testset "input validation" begin
        @test_throws ArgumentError jk_meanerror(Float64[])
        @test_throws ArgumentError jk_meanerror([1.0])
        @test_throws ArgumentError jk_meanerror(Float64[], mean)
        @test_throws ArgumentError jk_meanerror([1.0], mean)
        @test_throws ArgumentError jk_meanerror(Float64[], "mean")
        @test_throws ArgumentError jk_meanerror([1.0], "mean")
    end

    @testset "block jackknife indices" begin
        @test jk_block_index_set(1:6, 2) == [[3, 4, 5, 6], [1, 2, 5, 6], [1, 2, 3, 4]]
        @test jk_block_index_set(1:5, 2) == [[4, 5], [2, 3]]
        @test jk_block_index_set([2, 4, 6, 8, 10], 2) == [[8, 10], [4, 6]]

        @test jk_block_index([10.0, 20.0, 30.0, 40.0, 50.0], 2) == [[4, 5], [2, 3]]
        @test jk_block_index([10, 20, 30, 40, 50], 2) == [[4, 5], [2, 3]]

        @test_throws ArgumentError jk_block_index_set(1:4, 0)
        @test_throws ArgumentError jk_block_index_set(1:4, -1)
        @test_throws ArgumentError jk_block_index_set(1:4, 4)
        @test_throws ArgumentError jk_block_index_set(1:3, 5)
        @test_throws ArgumentError jk_block_index_set(1:4, true)
        @test_throws ArgumentError jk_block_index_set(1:4, 2.0)
        @test_throws ArgumentError jk_block_index([10.0, 20.0, 30.0, 40.0], true)
    end

    @testset "block mean jackknife error" begin
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        expected = block_jackknife_expected(data, 2, mean)

        test_result(jk_block_meanerror(data, 2), expected)
        test_result(jk_meanerror(data; block=2), expected)
        test_result(jk_block_meanerror(data, 2, "mean"), expected)
        test_result(jk_meanerror(data, "mean"; block=2), expected)
        test_result(jk_block_meanerror(data, 2, "average"), expected)
        test_result(jk_meanerror(data, "average"; block=2), expected)

        uneven_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        test_result(jk_block_meanerror(uneven_data, 2), block_jackknife_expected(uneven_data, 2, mean))
        test_result(jk_meanerror(uneven_data; block=2), block_jackknife_expected(uneven_data, 2, mean))
    end

    @testset "block observable keys" begin
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        susceptibility = x -> var(x, corrected=false)
        susceptibility_expected = block_jackknife_expected(data, 2, susceptibility)
        test_result(jk_block_meanerror(data, 2, "sus"), susceptibility_expected)
        test_result(jk_meanerror(data, "sus"; block=2), susceptibility_expected)
        test_result(jk_block_meanerror(data, 2, "susceptibility"), susceptibility_expected)
        test_result(jk_meanerror(data, "susceptibility"; block=2), susceptibility_expected)

        binder_ratio = x -> mean(x .^ 4) / mean(x .^ 2)^2
        binder_expected = block_jackknife_expected(data, 2, binder_ratio)
        test_result(jk_block_meanerror(data, 2, "binder"), binder_expected)
        test_result(jk_meanerror(data, "binder"; block=2), binder_expected)
        test_result(jk_block_meanerror(data, 2, "bin"), binder_expected)
        test_result(jk_meanerror(data, "bin"; block=2), binder_expected)

        @test_throws ArgumentError jk_block_meanerror(data, 2, "unknown")
        @test_throws ArgumentError jk_meanerror(data, "unknown"; block=2)
    end

    @testset "block custom statistic" begin
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        square_mean = x -> mean(x .^ 2)
        test_result(jk_block_meanerror(data, 2, square_mean), block_jackknife_expected(data, 2, square_mean))
        test_result(jk_meanerror(data, square_mean; block=2), block_jackknife_expected(data, 2, square_mean))

        susceptibility = x -> var(x, corrected=false)
        test_result(jk_block_meanerror(data, 2, susceptibility), block_jackknife_expected(data, 2, susceptibility))
        test_result(jk_meanerror(data, susceptibility; block=2), block_jackknife_expected(data, 2, susceptibility))
    end

    @testset "block real-valued inputs and validation" begin
        data = [1, 2, 3, 4, 5, 6]
        test_result(jk_block_meanerror(data, 2), block_jackknife_expected(data, 2, mean))
        test_result(jk_block_meanerror(data, 2, "sus"), block_jackknife_expected(data, 2, x -> var(x, corrected=false)))

        @test_throws ArgumentError jk_block_meanerror(Float64[], 1)
        @test_throws ArgumentError jk_block_meanerror([1.0], 1)
        @test_throws ArgumentError jk_block_meanerror([1.0, 2.0], 0)
        @test_throws ArgumentError jk_block_meanerror([1.0, 2.0], -1)
        @test_throws ArgumentError jk_block_meanerror([1.0, 2.0], 2)
        @test_throws ArgumentError jk_block_meanerror(Float64[], 1, mean)
        @test_throws ArgumentError jk_block_meanerror([1.0], 1, mean)
        @test_throws ArgumentError jk_block_meanerror([1.0, 2.0], 2, mean)
        @test_throws ArgumentError jk_block_meanerror(Float64[], 1, "mean")
        @test_throws ArgumentError jk_block_meanerror([1.0], 1, "mean")
        @test_throws ArgumentError jk_block_meanerror([1.0, 2.0], 2, "mean")
        @test_throws ArgumentError jk_meanerror([1.0, 2.0]; block=2)
        @test_throws ArgumentError jk_meanerror([1.0, 2.0]; block=2.0)
        @test_throws ArgumentError jk_meanerror([1.0, 2.0, 3.0, 4.0]; block=true)
        @test_throws ArgumentError jk_index([1.0, 2.0]; block=2.0)
        @test_throws ArgumentError jk_block_meanerror([1.0, 2.0, 3.0, 4.0], true)
        @test_throws ArgumentError jk_block_meanerror([1.0, 2.0, 3.0, 4.0], 2.0)
    end

    @testset "histogram jackknife errors" begin
        samples = [[0.2, 1.2], [0.8, 1.5], [1.1, 2.0]]
        edges = [0.0, 1.0, 2.0]

        hist = jk_histogram(samples, edges)
        @test hist.edges == edges
        @test hist.centers == [0.5, 1.5]
        @test hist.values ≈ [2 / 3, 4 / 3]
        @test hist.errors ≈ [1 / 3, 1 / 3]

        scaled = jk_histogram(samples, edges; scale=2.0, density=true)
        @test scaled.values ≈ [4 / 3, 8 / 3]
        @test scaled.errors ≈ [2 / 3, 2 / 3]

        flat = jk_histogram([0.2, 0.8, 1.5], edges)
        @test flat.values ≈ [2 / 3, 1 / 3]
        @test flat.errors ≈ [1 / 3, 1 / 3]
    end

    @testset "hist-like histogram API" begin
        samples = [[0.2, 1.2], [0.8, 1.5], [1.1, 2.0]]
        edges = [0.0, 1.0, 2.0]

        by_keyword = jk_histogram(samples; bins=edges)
        by_alias = jk_hist(samples; bins=edges)
        expected = jk_histogram(samples, edges)

        @test by_keyword == expected
        @test by_alias == expected

        auto_bins = jk_hist(samples; bins=2)
        @test auto_bins.edges ≈ [0.2, 1.1, 2.0]
        @test auto_bins.centers ≈ [0.65, 1.55]
        @test auto_bins.values ≈ [2 / 3, 4 / 3]
        @test auto_bins.errors ≈ [1 / 3, 1 / 3]
    end

    @testset "block histogram jackknife errors" begin
        samples = [[9.9], [0.2, 1.2], [0.4], [1.4], [1.6]]
        edges = [0.0, 1.0, 2.0]

        hist = jk_block_histogram(samples, edges, 2)
        @test hist.edges == edges
        @test hist.centers == [0.5, 1.5]
        @test hist.values ≈ [0.5, 0.75]
        @test hist.errors ≈ [0.5, 0.25]

        scaled = jk_block_histogram(samples, edges, 2; scale=2.0, density=true)
        @test scaled.values ≈ [1.0, 1.5]
        @test scaled.errors ≈ [1.0, 0.5]
    end

    @testset "hist-like block histogram API" begin
        samples = [[9.9], [0.2, 1.2], [0.4], [1.4], [1.6]]
        edges = [0.0, 1.0, 2.0]

        by_keyword = jk_block_histogram(samples, 2; bins=edges)
        by_alias = jk_block_hist(samples, 2; bins=edges)
        by_unified_keyword = jk_histogram(samples; bins=edges, block=2)
        by_unified_alias = jk_hist(samples; bins=edges, block=2)
        by_unified_edges = jk_histogram(samples, edges; block=2)
        expected = jk_block_histogram(samples, edges, 2)

        @test by_keyword == expected
        @test by_alias == expected
        @test by_unified_keyword == expected
        @test by_unified_alias == expected
        @test by_unified_edges == expected
    end

    @testset "histogram validation" begin
        samples = [[0.2, 1.2], [0.8, 1.5], [1.1, 2.0]]
        edges = [0.0, 1.0, 2.0]

        @test_throws ArgumentError jk_histogram([[0.1]], [0.0, 1.0])
        @test_throws ArgumentError jk_histogram(samples, [0.0])
        @test_throws ArgumentError jk_histogram(samples, [0.0, 1.0, 1.0])
        @test_throws ArgumentError jk_histogram(samples, [0.0, 2.0, 1.0])

        @test_throws ArgumentError jk_block_histogram([[0.1]], [0.0, 1.0], 1)
        @test_throws ArgumentError jk_block_histogram(samples, [0.0, 1.0], 0)
        @test_throws ArgumentError jk_block_histogram(samples, [0.0, 1.0], 3)
        @test_throws ArgumentError jk_block_histogram(samples, [0.0], 1)
        @test_throws ArgumentError jk_block_histogram(samples, [0.0, 1.0, 1.0], 1)

        @test_throws ArgumentError jk_hist(samples; bins=0)
        @test_throws ArgumentError jk_hist(samples; bins=-1)
        @test_throws ArgumentError jk_block_hist(samples, 1; bins=0)
        @test_throws ArgumentError jk_hist(samples; bins=edges, block=3)
        @test_throws ArgumentError jk_hist(samples; bins=edges, block=2.0)
        @test_throws ArgumentError jk_hist(samples; bins=edges, block=true)
        @test_throws ArgumentError jk_block_hist(samples, true; bins=edges)
        @test_throws ArgumentError jk_block_hist(samples, 2.0; bins=edges)
    end

    @testset "english help" begin
        help_text = sprint(jk_help)

        @test occursin("JackknifeTools quick help", help_text)
        @test occursin("Histogram with error bars", help_text)
        @test occursin("block=2", help_text)
        @test occursin("Binder ratio", help_text)
        @test occursin("mean(x .^ 4) / mean(x .^ 2)^2", help_text)
        @test !occursin("日本語", help_text)
        @test !occursin("クイックヘルプ", help_text)

        meanerror_doc = normalize_doc_whitespace(repr("text/plain", @doc jk_meanerror(::AbstractVector{<:Real}, ::String)))
        @test occursin("Binder ratio", meanerror_doc)
        @test occursin("mean(x .^ 4) / mean(x .^ 2)^2", meanerror_doc)

        doc_text = repr("text/plain", @doc jk_hist)
        @test occursin("Short alias", doc_text)
        @test !occursin("日本語", doc_text)
    end

    include("extensions.jl")

    @testset "tutorial smoke test" begin
        measurements = Ising2DTutorial.run_ising2d(;
            L=4,
            beta=0.3,
            therm_sweeps=4,
            sweeps=12,
            measure_every=1,
            seed=1234,
        )

        @test length(measurements.energies) == 12
        @test length(measurements.magnetizations) == 12

        analysis = Ising2DTutorial.analyze_measurements(measurements; block_size=2, hist_bins=4)

        @test analysis.block_size == 2
        @test isfinite(analysis.primary.energy[1])
        @test isfinite(analysis.secondary.susceptibility[1])
        @test isfinite(analysis.autocorrelation.abs_magnetization.tau_int)
        @test length(analysis.histogram.centers) == 4
        @test length(analysis.histogram.errors) == 4
        @test isfinite(analysis.secondary.susceptibility_signed[1])
        @test isfinite(analysis.autocorrelation.magnetization.tau_int)
        @test isfinite(analysis.multi.dabsm_dbeta[2])
        @test size(analysis.multi.means_cov) == (2, 2)
        @test Ising2DTutorial.suggested_block_size(11.132, 40_000) == 223
        @test Ising2DTutorial.suggested_block_size(11.1, 1_000) == 20
    end

    @testset "README quick start" begin
        data = randn(MersenneTwister(20), 10_000)
        A = 3 .+ data .+ 0.1 .* randn(MersenneTwister(21), 10_000)
        B = 2 .+ data .+ 0.1 .* randn(MersenneTwister(22), 10_000)

        @test length(jk_meanerror(data)) == 2
        @test length(jk_meanerror(data; block=50)) == 2
        @test length(jk_meanerror(data, x -> var(x, corrected=false))) == 2
        @test length(jk_meanerror(data, "binder"; block=50)) == 2
        ratio = jk_meanerror(A, B; block=50) do a, b
            mean(a) / mean(b)
        end
        @test ratio[1] ≈ mean(A) / mean(B)

        tau = jk_tau_int(data)
        @test tau.tau_int < 1
        scan = jk_block_scan(data; blocks=1:200)
        @test length(scan.block_sizes) == 200

        C = [exp(-0.3 * t) * (1 + 0.01 * randn(MersenneTwister(c))) for c in 1:40, t in 0:7]
        meff(c) = (ct = vec(mean(c; dims=1)); log.(ct[1:end-1] ./ ct[2:end]))
        m, m_err = jk_meanerror(C, meff; block=4)
        m2, m_cov = jk_meancov(C, meff; block=4)
        rep = jk_replicas(C, meff; block=4)
        @test m ≈ m2
        @test jk_error(rep.replicas) ≈ m_err
        @test jk_cov(rep.replicas) ≈ m_cov

        eigenvalues_by_config = [
            [0.002, 0.011, 0.018],
            [0.004, 0.010, 0.026],
            [0.001, 0.016, 0.021],
        ]
        hist = jk_hist(eigenvalues_by_config; bins=collect(0.0:0.01:0.03))
        @test hist.values ≈ [1.0, 4 / 3, 2 / 3]
        volume = 32^3 * 8
        hist2 = jk_hist(eigenvalues_by_config; bins=20, scale=1 / volume, density=true)
        @test length(hist2.values) == 20
    end
end
