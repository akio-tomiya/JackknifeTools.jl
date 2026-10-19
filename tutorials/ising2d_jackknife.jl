module Ising2DTutorial

using JackknifeTools
using Statistics

export analyze_measurements, integrated_autocorrelation_time, main, run_ising2d, suggested_block_size, write_summary

mutable struct XorShift64Star
    state::UInt64
end

function XorShift64Star(seed::Integer)
    state = UInt64(seed)
    state == 0 && (state = 0x9e3779b97f4a7c15)
    return XorShift64Star(state)
end

function rand_uint!(rng::XorShift64Star)
    x = rng.state
    x = xor(x, x >> 12)
    x = xor(x, x << 25)
    x = xor(x, x >> 27)
    rng.state = x
    return x * 0x2545f4914f6cdd1d
end

function rand_float!(rng::XorShift64Star)
    return Float64(rand_uint!(rng) >> 11) / 9007199254740992.0
end

function rand_int!(rng::XorShift64Star, n::Integer)
    n >= 1 || throw(ArgumentError("n must be positive"))
    return Int((rand_uint!(rng) >> 32) % UInt64(n)) + 1
end

function initial_spins(L::Integer, rng::XorShift64Star)
    L >= 2 || throw(ArgumentError("L must be at least 2"))
    spins = Matrix{Int8}(undef, L, L)
    for i in 1:L, j in 1:L
        spins[i, j] = rand_float!(rng) < 0.5 ? Int8(-1) : Int8(1)
    end
    return spins
end

function energy_per_spin(spins::AbstractMatrix{<:Integer})
    Lx, Ly = size(spins)
    Lx == Ly || throw(ArgumentError("spins must be a square matrix"))

    energy = 0
    for i in 1:Lx
        ip = i == Lx ? 1 : i + 1
        for j in 1:Ly
            jp = j == Ly ? 1 : j + 1
            energy -= spins[i, j] * (spins[ip, j] + spins[i, jp])
        end
    end

    return energy / length(spins)
end

function magnetization_per_spin(spins::AbstractMatrix{<:Integer})
    return sum(spins) / length(spins)
end

function heat_bath_sweep!(spins::AbstractMatrix{Int8}, beta::Real, rng::XorShift64Star)
    L = size(spins, 1)
    size(spins, 2) == L || throw(ArgumentError("spins must be a square matrix"))

    for _ in 1:length(spins)
        i = rand_int!(rng, L)
        j = rand_int!(rng, L)
        ip = i == L ? 1 : i + 1
        im = i == 1 ? L : i - 1
        jp = j == L ? 1 : j + 1
        jm = j == 1 ? L : j - 1

        neighbor_sum = spins[ip, j] + spins[im, j] + spins[i, jp] + spins[i, jm]
        probability_plus = 1 / (1 + exp(-2 * beta * neighbor_sum))

        spins[i, j] = rand_float!(rng) < probability_plus ? Int8(1) : Int8(-1)
    end

    return spins
end

function run_ising2d(;
    L::Integer=16,
    beta::Real=0.44,
    therm_sweeps::Integer=500,
    sweeps::Integer=200000,
    measure_every::Integer=5,
    seed::Integer=20260515,
)
    L >= 2 || throw(ArgumentError("L must be at least 2"))
    beta > 0 || throw(ArgumentError("beta must be positive"))
    therm_sweeps >= 0 || throw(ArgumentError("therm_sweeps must be non-negative"))
    sweeps >= 1 || throw(ArgumentError("sweeps must be positive"))
    measure_every >= 1 || throw(ArgumentError("measure_every must be positive"))
    seed >= 0 || throw(ArgumentError("seed must be non-negative"))

    rng = XorShift64Star(seed)
    spins = initial_spins(L, rng)

    for _ in 1:therm_sweeps
        heat_bath_sweep!(spins, beta, rng)
    end

    nmeasure = div(sweeps, measure_every)
    nmeasure >= 2 || throw(ArgumentError("at least two measurements are required"))

    energies = Float64[]
    magnetizations = Float64[]
    sizehint!(energies, nmeasure)
    sizehint!(magnetizations, nmeasure)

    for sweep in 1:sweeps
        heat_bath_sweep!(spins, beta, rng)
        if sweep % measure_every == 0
            push!(energies, energy_per_spin(spins))
            push!(magnetizations, magnetization_per_spin(spins))
        end
    end

    return (
        L=Int(L),
        beta=Float64(beta),
        therm_sweeps=Int(therm_sweeps),
        sweeps=Int(sweeps),
        measure_every=Int(measure_every),
        energies=energies,
        magnetizations=magnetizations,
        abs_magnetizations=abs.(magnetizations),
    )
end

function integrated_autocorrelation_time(series::AbstractVector{<:Real}; kwargs...)
    return jk_tau_int(series; kwargs...)
end

# The block Jackknife error of a mean is biased low by a relative amount of
# order tau_int / b, so the block size should be many times tau_int. The
# default takes 20 tau_int and keeps at least 50 blocks. Check the choice with
# jk_block_scan.
function suggested_block_size(tau_int::Real, nsamples::Integer)
    nsamples >= 2 || throw(ArgumentError("at least two samples are required"))
    max_block_size = max(1, div(nsamples, 50))
    return min(max(2, ceil(Int, 20 * tau_int)), max_block_size)
end

function analyze_measurements(measurements; block_size=nothing, hist_bins::Integer=12)
    energies = measurements.energies
    magnetizations = measurements.magnetizations
    abs_magnetizations = measurements.abs_magnetizations
    nspins = measurements.L^2
    beta = measurements.beta

    tau_abs_m = jk_tau_int(abs_magnetizations)
    tau_energy = jk_tau_int(energies)
    tau_m = jk_tau_int(magnetizations)
    chosen_block_size = block_size === nothing ?
        suggested_block_size(max(tau_abs_m.tau_int, tau_energy.tau_int), length(abs_magnetizations)) :
        block_size

    # chi' = N beta (<m^2> - <|m|>^2) uses |m|. The signed version
    # N beta (<m^2> - <m>^2) depends on rare sign flips of m; it is kept only
    # to show that problem.
    susceptibility = x -> nspins * beta * var(x, corrected=false)
    specific_heat = x -> nspins * beta^2 * var(x, corrected=false)

    # d<|m|>/d beta = -N_spin (<|m| e> - <|m|><e>), built from two series
    # measured on the same configurations.
    dabsm_dbeta = (m, e) -> -nspins * (mean(m .* e) - mean(m) * mean(e))

    # The same quantity with the three averages treated as independent. This
    # ignores the correlations and is shown only for comparison.
    me = abs_magnetizations .* energies
    me_err = jk_meanerror(me; block=chosen_block_size)[2]
    m_err = jk_meanerror(abs_magnetizations; block=chosen_block_size)[2]
    e_err = jk_meanerror(energies; block=chosen_block_size)[2]
    naive_dabsm_dbeta_error = nspins * sqrt(me_err^2 + (mean(energies) * m_err)^2 + (mean(abs_magnetizations) * e_err)^2)

    two_means = (m, e) -> [mean(m), mean(e)]
    means_central, means_cov = jk_meancov(two_means, abs_magnetizations, energies; block=chosen_block_size)

    return (
        primary=(
            energy=jk_meanerror(energies),
            abs_magnetization=jk_meanerror(abs_magnetizations),
        ),
        secondary=(
            susceptibility=jk_meanerror(abs_magnetizations, susceptibility),
            susceptibility_signed=jk_meanerror(magnetizations, susceptibility),
            specific_heat=jk_meanerror(energies, specific_heat),
            binder_ratio=jk_meanerror(magnetizations, "binder"),
        ),
        autocorrelation=(
            abs_magnetization=tau_abs_m,
            energy=tau_energy,
            magnetization=tau_m,
        ),
        block_size=chosen_block_size,
        block=(
            energy=jk_meanerror(energies; block=chosen_block_size),
            abs_magnetization=jk_meanerror(abs_magnetizations; block=chosen_block_size),
            susceptibility=jk_meanerror(abs_magnetizations, susceptibility; block=chosen_block_size),
            susceptibility_signed=jk_meanerror(magnetizations, susceptibility; block=chosen_block_size),
            specific_heat=jk_meanerror(energies, specific_heat; block=chosen_block_size),
            binder_ratio=jk_meanerror(magnetizations, "binder"; block=chosen_block_size),
        ),
        multi=(
            dabsm_dbeta=jk_meanerror(dabsm_dbeta, abs_magnetizations, energies; block=chosen_block_size),
            naive_dabsm_dbeta_error=naive_dabsm_dbeta_error,
            means=means_central,
            means_cov=means_cov,
            correlation=means_cov[1, 2] / sqrt(means_cov[1, 1] * means_cov[2, 2]),
        ),
        histogram=jk_hist(abs_magnetizations; bins=hist_bins, block=chosen_block_size, density=true),
    )
end

function fmt(x::Real; digits::Integer=8)
    return string(round(Float64(x); digits=digits))
end

function print_estimate(io::IO, label::AbstractString, estimate)
    println(io, rpad(label, 32), " ", fmt(estimate[1]), " +/- ", fmt(estimate[2]))
end

function print_estimate(label::AbstractString, estimate)
    return print_estimate(stdout, label, estimate)
end

function print_histogram(io::IO, hist; max_rows::Integer=8)
    rows = min(max_rows, length(hist.centers))
    println(io, "center         density        error")
    for i in 1:rows
        println(io, rpad(fmt(hist.centers[i]), 15), rpad(fmt(hist.values[i]), 15), fmt(hist.errors[i]))
    end
    length(hist.centers) > rows && println(io, "...")
end

function print_histogram(hist; max_rows::Integer=8)
    return print_histogram(stdout, hist; max_rows=max_rows)
end

function write_summary(io::IO, measurements, analysis)
    println(io, "2D Ising Jackknife tutorial")
    println(io, "L = $(measurements.L), beta = $(measurements.beta)")
    println(io, "update = heat bath")
    println(io, "measurements = $(length(measurements.energies)), measure_every = $(measurements.measure_every)")
    println(io)

    println(io, "Ordinary Jackknife: primary observables")
    print_estimate(io, "energy per spin", analysis.primary.energy)
    print_estimate(io, "abs magnetization per spin", analysis.primary.abs_magnetization)
    println(io)

    println(io, "Ordinary Jackknife: secondary observables")
    print_estimate(io, "susceptibility chi'", analysis.secondary.susceptibility)
    print_estimate(io, "chi from signed m", analysis.secondary.susceptibility_signed)
    print_estimate(io, "specific heat", analysis.secondary.specific_heat)
    print_estimate(io, "Binder ratio", analysis.secondary.binder_ratio)
    println(io)

    println(io, "Integrated autocorrelation time (automatic window, c = 6)")
    for (label, tau) in (
        ("|m|", analysis.autocorrelation.abs_magnetization),
        ("energy", analysis.autocorrelation.energy),
        ("signed m", analysis.autocorrelation.magnetization),
    )
        println(io, rpad("tau_int of $label", 32), " ", fmt(tau.tau_int; digits=3), " +/- ", fmt(tau.error; digits=3),
            " measurement intervals (window = ", tau.window, ")")
    end
    println(io, "Using block_size = $(analysis.block_size) (20 tau_int of |m| and energy)")
    println(io)

    println(io, "Block Jackknife")
    print_estimate(io, "energy per spin", analysis.block.energy)
    print_estimate(io, "abs magnetization per spin", analysis.block.abs_magnetization)
    print_estimate(io, "susceptibility chi'", analysis.block.susceptibility)
    print_estimate(io, "chi from signed m", analysis.block.susceptibility_signed)
    print_estimate(io, "specific heat", analysis.block.specific_heat)
    print_estimate(io, "Binder ratio", analysis.block.binder_ratio)
    println(io)

    println(io, "Two correlated series: d<|m|>/d beta = -N_spin (<|m| e> - <|m|><e>)")
    print_estimate(io, "block Jackknife", analysis.multi.dabsm_dbeta)
    println(io, rpad("error if treated as independent", 32), " ", fmt(analysis.multi.naive_dabsm_dbeta_error))
    println(io, rpad("correlation of <|m|> and <e>", 32), " ", fmt(analysis.multi.correlation; digits=4))
    println(io)

    println(io, "Histogram of |m| with block Jackknife error bars")
    print_histogram(io, analysis.histogram)

    return nothing
end

function main()
    measurements = run_ising2d()
    analysis = analyze_measurements(measurements)
    write_summary(stdout, measurements, analysis)
    return nothing
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    Ising2DTutorial.main()
end
