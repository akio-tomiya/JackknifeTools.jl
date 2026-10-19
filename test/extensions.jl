# Tests for the O(N) estimators, several series, vector and complex
# observables, histograms with non-finite values, and autocorrelation tools.

# Explicit leave-one-block-out samples for several series, written
# independently from the package (same convention as the handoff note).
function multi_block_expected(f, series::Tuple, block_size::Integer)
    n = length(first(series))
    first_kept = mod(n, block_size) + 1
    starts = first_kept:block_size:n
    central = f(map(x -> x[first_kept:n], series)...)
    replicas = [
        f(map(x -> x[vcat(first_kept:s-1, s+block_size:n)], series)...)
        for s in starts
    ]
    nb = length(starts)
    return central, replicas, nb
end

function ar1_series(rng, n, phi)
    x = zeros(n)
    x[1] = randn(rng) / sqrt(1 - phi^2)
    for i in 2:n
        x[i] = phi * x[i-1] + randn(rng)
    end
    return x
end

ar1_tau_int(phi) = (1 + phi) / (2 * (1 - phi))

diag_of(m) = [m[i, i] for i in 1:size(m, 1)]

@testset "O(N) mean and replicas" begin
    rng = MersenneTwister(11)
    x = randn(rng, 101)

    # Mean: fast path agrees with the generic leave-out path and with the
    # textbook formula std(x) / sqrt(N).
    fast = jk_meanerror(x)
    generic = jk_meanerror(x, s -> sum(s) / length(s))
    @test fast[1] == mean(x)
    @test fast[1] ≈ generic[1]
    @test fast[2] ≈ generic[2]
    @test fast[2] ≈ std(x) / sqrt(length(x))

    for b in (1, 2, 5, 7)
        fast_b = jk_meanerror(x; block=b)
        generic_b = jk_meanerror(x, s -> sum(s) / length(s); block=b)
        first_kept = mod(length(x), b) + 1
        block_means = [mean(x[s:s+b-1]) for s in first_kept:b:length(x)]
        @test fast_b[1] ≈ generic_b[1]
        @test fast_b[2] ≈ generic_b[2]
        @test fast_b[2] ≈ std(block_means) / sqrt(length(block_means))
    end

    rep = jk_replicas(x, "mean"; block=4)
    @test length(rep.replicas) == div(length(x), 4)
    @test jk_error(rep.replicas) ≈ jk_meanerror(x; block=4)[2]
    @test rep.central ≈ jk_meanerror(x; block=4)[1]

    # Replicas of a nonlinear statistic match explicit subsets.
    sus = s -> var(s, corrected=false)
    rep_sus = jk_replicas(x, sus; block=3)
    expected_central, expected_replicas, _ = multi_block_expected(sus, (x,), 3)
    @test rep_sus.central ≈ expected_central
    @test rep_sus.replicas ≈ expected_replicas

    # Results that alias the reused buffer are copied.
    rep_view = jk_replicas(collect(1.0:6.0), s -> view(s, 1:2); block=2)
    @test rep_view.replicas == [[3.0, 4.0], [1.0, 2.0], [1.0, 2.0]]
end

@testset "O(N) memory" begin
    rng = MersenneTwister(12)
    long = randn(rng, 20_000)
    jk_meanerror(long)
    jk_meanerror(long; block=10)
    @test (@allocated jk_meanerror(long)) < 2_000_000
    @test (@allocated jk_meanerror(long; block=10)) < 2_000_000

    medium = randn(rng, 5_000)
    f = s -> sum(abs2, s) / length(s)
    jk_meanerror(medium, f)
    @test (@allocated jk_meanerror(medium, f)) < 5_000_000

    hist_samples = randn(rng, 20_000)
    edges = collect(-4.0:0.5:4.0)
    jk_hist(hist_samples; bins=edges)
    @test (@allocated jk_hist(hist_samples; bins=edges)) < 40_000_000
end

@testset "several correlated series" begin
    rng = MersenneTwister(13)
    n = 203
    common = randn(rng, n)
    A = 3.0 .+ common .+ 0.1 .* randn(rng, n)
    B = 2.0 .+ common .+ 0.1 .* randn(rng, n)
    ratio = (a, b) -> mean(a) / mean(b)

    for b in (1, 4, 10)
        central, replicas, nb = multi_block_expected(ratio, (A, B), b)
        expected_error = std(replicas, corrected=false) * sqrt(nb - 1)

        got = jk_meanerror(ratio, A, B; block=b)
        @test got[1] ≈ central
        @test got[2] ≈ expected_error

        got_do = jk_meanerror(A, B; block=b) do a, c
            mean(a) / mean(c)
        end
        @test got_do[1] ≈ got[1]
        @test got_do[2] ≈ got[2]
    end

    # Ignoring the correlation between A and B would overestimate this error.
    err_a = jk_meanerror(A)[2]
    err_b = jk_meanerror(B)[2]
    naive = abs(mean(A) / mean(B)) * sqrt((err_a / mean(A))^2 + (err_b / mean(B))^2)
    @test jk_meanerror(ratio, A, B)[2] < naive / 2

    # A linear combination of series has the same error as its sample series.
    lin = jk_meanerror((a, c) -> 2 * mean(a) - 3 * mean(c), A, B; block=4)
    direct = jk_meanerror(2 .* A .- 3 .* B; block=4)
    @test lin[1] ≈ direct[1]
    @test lin[2] ≈ direct[2]

    @test_throws ArgumentError jk_meanerror(ratio, A, B[1:end-1])
    @test_throws ArgumentError jk_replicas(ratio, A, B[1:end-1])
end

@testset "vector-valued observables and covariance" begin
    rng = MersenneTwister(14)
    x = randn(rng, 120)
    moments = s -> [mean(s), mean(s .^ 2), mean(s .^ 4)]

    value, error = jk_meanerror(x, moments; block=3)
    for (k, f) in enumerate((mean, s -> mean(s .^ 2), s -> mean(s .^ 4)))
        scalar = jk_meanerror(x, f; block=3)
        @test value[k] ≈ scalar[1]
        @test error[k] ≈ scalar[2]
    end

    central, cov = jk_meancov(x, moments; block=3)
    @test central ≈ value
    @test size(cov) == (3, 3)
    @test cov ≈ cov'
    @test sqrt.(diag_of(cov)) ≈ error
    @test jk_cov(jk_replicas(x, moments; block=3).replicas) ≈ cov

    scalar_central, scalar_cov = jk_meancov(x; block=3)
    @test scalar_central ≈ mean(x[mod(120, 3)+1:end])
    @test scalar_cov[1, 1] ≈ jk_meanerror(x; block=3)[2]^2

    @test_throws ArgumentError jk_cov([1.0])
    @test_throws ArgumentError jk_cov([[1.0, 2.0], [1.0]])
    @test_throws ArgumentError jk_cov([1.0 + 1.0im, 2.0 + 0.0im])
end

@testset "matrix input with rows as samples" begin
    rng = MersenneTwister(15)
    nconf, nt = 64, 6
    masses = 0.5
    C = [exp(-masses * t) * (1 + 0.05 * randn(rng)) for _ in 1:nconf, t in 0:nt-1]

    value, error = jk_meanerror(C; block=4)
    for t in 1:nt
        column = jk_meanerror(C[:, t]; block=4)
        @test value[t] ≈ column[1]
        @test error[t] ≈ column[2]
    end

    meff(c) = (ct = vec(mean(c; dims=1)); log.(ct[1:end-1] ./ ct[2:end]))
    m, m_err = jk_meanerror(C, meff; block=4)
    @test length(m) == nt - 1
    @test all(abs.(m .- masses) .< 5 .* m_err)

    rows = [C[i, :] for i in 1:nconf]
    m_rows, m_rows_err = jk_meanerror(rows, r -> (ct = mean(r); log.(ct[1:end-1] ./ ct[2:end])); block=4)
    @test m_rows ≈ m
    @test m_rows_err ≈ m_err

    m_central, m_cov = jk_meancov(C, meff; block=4)
    @test m_central ≈ m
    @test sqrt.(diag_of(m_cov)) ≈ m_err
end

@testset "complex observables" begin
    rng = MersenneTwister(16)
    z = randn(rng, ComplexF64, 50)
    value, error = jk_meanerror(z; block=2)
    @test real(value) ≈ jk_meanerror(real.(z); block=2)[1]
    @test imag(value) ≈ jk_meanerror(imag.(z); block=2)[1]
    @test real(error) ≈ jk_meanerror(real.(z); block=2)[2]
    @test imag(error) ≈ jk_meanerror(imag.(z); block=2)[2]

    phase = s -> mean(s) / abs(mean(s))
    value_phase, error_phase = jk_meanerror(z, phase)
    @test abs(value_phase) ≈ 1
    @test real(error_phase) >= 0 && imag(error_phase) >= 0
end

@testset "observable keys as strings and symbols" begin
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    expected = jk_meanerror(data, "sus"; block=2)
    @test jk_meanerror(data, :sus; block=2) == expected
    @test jk_meanerror(data, split("sus,mean", ",")[1]; block=2) == expected
    @test jk_block_meanerror(data, 2, :binder) == jk_meanerror(data, "binder"; block=2)
    @test_throws ArgumentError jk_meanerror(data, :unknown)
end

@testset "histograms with non-finite values" begin
    samples = [[0.2, NaN], [0.8, Inf], [1.5, -Inf], [1.9]]
    hist = jk_hist(samples; bins=[0.0, 1.0, 2.0])
    clean = jk_hist([[0.2], [0.8], [1.5], [1.9]]; bins=[0.0, 1.0, 2.0])
    @test hist == clean

    auto = jk_hist([1.0, NaN, 2.0, Inf, 3.0]; bins=2)
    @test auto.edges ≈ [1.0, 2.0, 3.0]

    @test_throws ArgumentError jk_hist([NaN, Inf]; bins=2)
    @test_throws ArgumentError jk_hist([1.0, 2.0]; bins=true)
    @test_throws ArgumentError jk_hist([1.0, 2.0]; bins=2.5)
    @test_throws ArgumentError jk_hist([1.0, 2.0]; bins=[0.0, Inf])

    # Histogram errors agree with the generic Jackknife of each bin count.
    rng = MersenneTwister(17)
    values = [randn(rng, 3) for _ in 1:30]
    edges = [-3.0, -1.0, 0.0, 1.0, 3.0]
    h = jk_hist(values; bins=edges, block=3)
    for k in 1:length(edges)-1
        in_bin = v -> count(x -> edges[k] <= x < edges[k+1] || (k == length(edges) - 1 && x == edges[end]), v)
        counts = Float64[in_bin(v) for v in values]
        expected = jk_meanerror(counts; block=3)
        @test h.values[k] ≈ expected[1]
        @test h.errors[k] ≈ expected[2]
    end
end

@testset "autocorrelation and tau_int" begin
    @test jk_autocorr([1.0, 2.0, 3.0, 4.0], 0) == [1.0]
    @test jk_autocorr(fill(2.0, 5), 3) == [1.0, 0.0, 0.0, 0.0]
    @test_throws ArgumentError jk_autocorr([1.0], 0)
    @test_throws ArgumentError jk_autocorr([1.0, 2.0], 2)

    rng = MersenneTwister(18)
    phi = 0.8
    x = ar1_series(rng, 100_000, phi)
    exact = ar1_tau_int(phi)  # 4.5

    rho = jk_autocorr(x, 5)
    @test all(abs.(rho .- phi .^ (0:5)) .< 0.02)

    tau = jk_tau_int(x)
    @test tau.window >= 6 * tau.tau_int
    @test abs(tau.tau_int - exact) < 3 * tau.error
    @test tau.error > 0
    @test length(tau.rho) == length(x) ÷ 2 + 1

    fixed = jk_tau_int(x; window=10, max_lag=50)
    @test fixed.window == 10
    @test fixed.tau_int ≈ 0.5 + sum(fixed.rho[2:11])

    first_nonpositive = jk_tau_int(x; window=:first_nonpositive, max_lag=200)
    W = first_nonpositive.window
    @test all(first_nonpositive.rho[2:W+1] .> 0)
    @test W == 200 || first_nonpositive.rho[W+2] <= 0

    @test jk_tau_int(randn(rng, 10_000)).tau_int < 0.7
    @test_throws ArgumentError jk_tau_int(x; window=:unknown)
    @test_throws ArgumentError jk_tau_int(x; window=-1)
    @test_throws ArgumentError jk_tau_int(x; window=true)
    @test_throws ArgumentError jk_tau_int(x; c=0)
end

@testset "block scan" begin
    rng = MersenneTwister(19)
    phi = 0.8
    x = ar1_series(rng, 50_000, phi)

    scan = jk_block_scan(x; blocks=[1, 2, 5, 100, 200, 400, 30_000])
    @test scan.block_sizes == [1, 2, 5, 100, 200, 400]
    @test scan.nblocks == div.(50_000, scan.block_sizes)
    @test scan.errors[1] ≈ jk_meanerror(x)[2]
    @test scan.values[4] ≈ jk_meanerror(x; block=100)[1]

    # For b ≫ τ_int the error ratio approaches sqrt(2 τ_int) = 3.
    ratio = scan.errors[end-1] / scan.errors[1]
    @test abs(ratio - sqrt(2 * ar1_tau_int(phi))) < 0.5

    default_scan = jk_block_scan(x[1:200])
    @test default_scan.block_sizes == collect(1:20)

    scan_key = jk_block_scan(x[1:300], :sus; blocks=1:3)
    @test scan_key.values ≈ [jk_meanerror(x[1:300], "sus"; block=b)[1] for b in 1:3]

    A = x[1:300]
    B = x[1:300] .^ 2 .+ 1
    scan_multi = jk_block_scan((a, b) -> mean(a) / mean(b), A, B; blocks=[1, 3])
    @test scan_multi.errors[2] ≈ jk_meanerror((a, b) -> mean(a) / mean(b), A, B; block=3)[2]

    @test_throws ArgumentError jk_block_scan(x[1:10]; blocks=[6, 10])
    @test_throws ArgumentError jk_block_scan(x[1:10]; blocks=[0])
end

@testset "Aqua" begin
    Aqua.test_all(JackknifeTools)
end
