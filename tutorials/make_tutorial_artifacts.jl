include(joinpath(@__DIR__, "ising2d_jackknife.jl"))

using .Ising2DTutorial
using JackknifeTools

const ROOT = @__DIR__
const FIGURE_DIR = joinpath(ROOT, "figures")
const RESULT_DIR = joinpath(ROOT, "results")

function svg_escape(text)
    s = string(text)
    s = replace(s, "&" => "&amp;")
    s = replace(s, "<" => "&lt;")
    s = replace(s, ">" => "&gt;")
    return replace(s, "\"" => "&quot;")
end

function svg_header(width::Integer, height::Integer)
    return """
    <svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">
      <rect width="100%" height="100%" fill="white"/>
      <style>
        text { font-family: Arial, Helvetica, sans-serif; fill: #222; }
        .title { font-size: 20px; font-weight: bold; }
        .label { font-size: 15px; }
        .tick { font-size: 13px; fill: #555; }
        .axis { stroke: #222; stroke-width: 1.2; }
        .grid { stroke: #ddd; stroke-width: 1; }
      </style>
    """
end

function svg_footer()
    return "</svg>\n"
end

function data_range(values; include_zero::Bool=false)
    ymin = minimum(values)
    ymax = maximum(values)
    if include_zero
        ymin = min(ymin, 0.0)
        ymax = max(ymax, 0.0)
    end
    if ymin == ymax
        delta = ymin == 0 ? 1.0 : abs(ymin) * 0.1
        ymin -= delta
        ymax += delta
    end
    pad = 0.08 * (ymax - ymin)
    return ymin - pad, ymax + pad
end

function sx(x, xmin, xmax, left, width)
    return left + (x - xmin) / (xmax - xmin) * width
end

function sy(y, ymin, ymax, top, height)
    return top + (ymax - y) / (ymax - ymin) * height
end

function path_points(xs, ys, xmin, xmax, ymin, ymax, left, top, width, height)
    parts = String[]
    for (i, (x, y)) in enumerate(zip(xs, ys))
        command = i == 1 ? "M" : "L"
        push!(parts, "$command $(round(sx(x, xmin, xmax, left, width); digits=2)) $(round(sy(y, ymin, ymax, top, height); digits=2))")
    end
    return join(parts, " ")
end

# Tick positions at round values: steps of 1, 2, 2.5, or 5 times a power of 10.
function nice_ticks(vmin, vmax; target=5)
    span = vmax - vmin
    span > 0 || return [vmin], 1.0
    raw = span / target
    magnitude = 10.0^floor(log10(raw))
    step = magnitude * first(m for m in (1.0, 2.0, 2.5, 5.0, 10.0) if m * magnitude >= raw)
    first_tick = ceil(vmin / step) * step
    ticks = collect(first_tick:step:vmax + 1e-9 * span)
    return ticks, step
end

function tick_label(value, step)
    digits = max(0, -floor(Int, log10(step) + 1e-9) + (step / 10.0^floor(log10(step)) == 2.5 ? 1 : 0))
    abs(value) < 1e-12 * max(1.0, abs(step)) && (value = 0.0)
    return digits == 0 ? string(round(Int, value)) : string(round(value; digits=digits))
end

function draw_axes(io, left, top, width, height, xmin, xmax, ymin, ymax; xlabel="", ylabel="", draw_x_tick_labels=true)
    right = left + width
    bottom = top + height

    println(io, """<line class="axis" x1="$left" y1="$bottom" x2="$right" y2="$bottom"/>""")
    println(io, """<line class="axis" x1="$left" y1="$top" x2="$left" y2="$bottom"/>""")

    if draw_x_tick_labels
        xticks, xstep = nice_ticks(xmin, xmax)
        for value in xticks
            x = sx(value, xmin, xmax, left, width)
            println(io, """<line class="grid" x1="$(round(x; digits=2))" y1="$top" x2="$(round(x; digits=2))" y2="$bottom"/>""")
            println(io, """<text class="tick" x="$(round(x; digits=2))" y="$(bottom + 20)" text-anchor="middle">$(tick_label(value, xstep))</text>""")
        end
    end

    yticks, ystep = nice_ticks(ymin, ymax)
    for value in yticks
        y = sy(value, ymin, ymax, top, height)
        println(io, """<line class="grid" x1="$left" y1="$(round(y; digits=2))" x2="$right" y2="$(round(y; digits=2))"/>""")
        println(io, """<text class="tick" x="$(left - 8)" y="$(round(y + 4; digits=2))" text-anchor="end">$(tick_label(value, ystep))</text>""")
    end

    if !isempty(xlabel)
        println(io, """<text class="label" x="$(left + width / 2)" y="$(bottom + 44)" text-anchor="middle">$(svg_escape(xlabel))</text>""")
    end
    if !isempty(ylabel)
        println(io, """<text class="label" transform="translate($(left - 58), $(top + height / 2)) rotate(-90)" text-anchor="middle">$(svg_escape(ylabel))</text>""")
    end
end

function write_time_series_svg(path, measurements)
    width = 900
    height = 540
    nshow = min(1500, length(measurements.energies))
    xs = collect(1:nshow)

    open(path, "w") do io
        print(io, svg_header(width, height))
        println(io, """<text class="title" x="450" y="32" text-anchor="middle">2D Ising heat-bath Monte Carlo time series (first $nshow measurements)</text>""")

        panels = [
            (top=70, values=measurements.energies[1:nshow], ylabel="energy per spin", color="#20639b"),
            (top=315, values=measurements.abs_magnetizations[1:nshow], ylabel="|m|", color="#b23a48"),
        ]

        for panel in panels
            left = 90
            top = panel.top
            plot_width = 760
            plot_height = 170
            ymin, ymax = data_range(panel.values)
            xmin, xmax = 1.0, Float64(length(xs))
            draw_axes(io, left, top, plot_width, plot_height, xmin, xmax, ymin, ymax; xlabel="measurement index", ylabel=panel.ylabel)
            d = path_points(xs, panel.values, xmin, xmax, ymin, ymax, left, top, plot_width, plot_height)
            println(io, """<path d="$d" fill="none" stroke="$(panel.color)" stroke-width="1.5"/>""")
        end

        print(io, svg_footer())
    end
end

function write_autocorrelation_svg(path, analysis)
    width = 760
    height = 480
    rho = analysis.autocorrelation.abs_magnetization.rho
    max_lag = min(length(rho) - 1, 80)
    xs = collect(0:max_lag)
    ys = rho[1:max_lag+1]
    ymin, ymax = data_range(vcat(ys, [0.0, 1.0]))

    open(path, "w") do io
        print(io, svg_header(width, height))
        println(io, """<text class="title" x="380" y="32" text-anchor="middle">Autocorrelation function of |m|</text>""")
        left, top, plot_width, plot_height = 85, 70, 620, 330
        draw_axes(io, left, top, plot_width, plot_height, 0.0, Float64(max_lag), ymin, ymax; xlabel="lag", ylabel="rho(t)")
        zero_y = sy(0.0, ymin, ymax, top, plot_height)
        println(io, """<line x1="$left" y1="$(round(zero_y; digits=2))" x2="$(left + plot_width)" y2="$(round(zero_y; digits=2))" stroke="#888" stroke-dasharray="5,5"/>""")
        d = path_points(xs, ys, 0.0, Float64(max_lag), ymin, ymax, left, top, plot_width, plot_height)
        println(io, """<path d="$d" fill="none" stroke="#20639b" stroke-width="2"/>""")
        tau_result = analysis.autocorrelation.abs_magnetization
        tau = round(tau_result.tau_int; digits=2)
        tau_err = round(tau_result.error; digits=2)
        window = tau_result.window
        if window <= max_lag
            wx = sx(window, 0.0, Float64(max_lag), left, plot_width)
            println(io, """<line x1="$(round(wx; digits=2))" y1="$top" x2="$(round(wx; digits=2))" y2="$(top + plot_height)" stroke="#b23a48" stroke-dasharray="5,5"/>""")
        end
        println(io, """<text class="label" x="$(left + plot_width - 10)" y="93" text-anchor="end">tau_int = $tau +/- $tau_err, automatic window W = $window</text>""")
        print(io, svg_footer())
    end
end

function draw_box_row(io, labels, omitted, y; row_label, x0=220, box_width=54, box_height=34, gap=14, colors=nothing, dropped=Int[])
    println(io, """<text class="label" x="$(x0 - 25)" y="$(y + 22)" text-anchor="end">$(svg_escape(row_label))</text>""")
    for (i, label) in enumerate(labels)
        x = x0 + (i - 1) * (box_width + gap)
        is_omitted = i in omitted
        is_dropped = i in dropped
        color = colors === nothing ? "#6aaed6" : colors[i]
        fill = is_dropped ? "#e6e6e6" : is_omitted ? "#f5f5f5" : color
        stroke = is_omitted ? "#777" : "#333"
        dash = is_omitted ? " stroke-dasharray=\"5,4\"" : ""
        opacity = is_dropped ? "0.85" : "1.0"
        println(io, """<rect x="$x" y="$y" width="$box_width" height="$box_height" rx="4" fill="$fill" opacity="$opacity" stroke="$stroke" stroke-width="1.2"$dash/>""")
        println(io, """<text class="label" x="$(x + box_width / 2)" y="$(y + 22)" text-anchor="middle">$(svg_escape(label))</text>""")
        if is_omitted
            println(io, """<line x1="$(x + 10)" y1="$(y + 8)" x2="$(x + box_width - 10)" y2="$(y + box_height - 8)" stroke="#777" stroke-width="1.6"/>""")
            println(io, """<line x1="$(x + box_width - 10)" y1="$(y + 8)" x2="$(x + 10)" y2="$(y + box_height - 8)" stroke="#777" stroke-width="1.6"/>""")
        end
    end
end

function write_jackknife_samples_svg(path)
    width = 820
    height = 445
    labels = ["x1", "x2", "x3", "x4", "x5"]

    open(path, "w") do io
        print(io, svg_header(width, height))
        println(io, """<text class="title" x="410" y="32" text-anchor="middle">Ordinary Jackknife: leave one measurement out</text>""")
        println(io, """<text class="label" x="410" y="62" text-anchor="middle">Each Jackknife sample removes one data point and keeps the rest.</text>""")

        draw_box_row(io, labels, Int[], 92; row_label="full data")
        for i in 1:length(labels)
            draw_box_row(io, labels, [i], 92 + i * 52; row_label="sample $i")
        end

        println(io, """<text class="tick" x="410" y="428" text-anchor="middle">The estimator is evaluated on each leave-one-out sample.</text>""")
        print(io, svg_footer())
    end
end

function write_histogram_svg(path, hist)
    width = 760
    height = 480
    ymin = 0.0
    ymax = 1.12 * maximum(hist.values .+ hist.errors)

    open(path, "w") do io
        print(io, svg_header(width, height))
        println(io, """<text class="title" x="380" y="32" text-anchor="middle">Histogram of |m| with block Jackknife errors</text>""")
        left, top, plot_width, plot_height = 85, 70, 620, 330
        xmin, xmax = minimum(hist.edges), maximum(hist.edges)
        draw_axes(io, left, top, plot_width, plot_height, xmin, xmax, ymin, ymax; xlabel="|m|", ylabel="density")

        for i in eachindex(hist.centers)
            x1 = sx(hist.edges[i], xmin, xmax, left, plot_width)
            x2 = sx(hist.edges[i+1], xmin, xmax, left, plot_width)
            y = sy(hist.values[i], ymin, ymax, top, plot_height)
            bottom = sy(0.0, ymin, ymax, top, plot_height)
            bar_width = max(1.0, x2 - x1 - 2)
            println(io, """<rect x="$(round(x1 + 1; digits=2))" y="$(round(y; digits=2))" width="$(round(bar_width; digits=2))" height="$(round(bottom - y; digits=2))" fill="#6aaed6" opacity="0.75"/>""")

            cx = sx(hist.centers[i], xmin, xmax, left, plot_width)
            y_low = sy(hist.values[i] - hist.errors[i], ymin, ymax, top, plot_height)
            y_high = sy(hist.values[i] + hist.errors[i], ymin, ymax, top, plot_height)
            println(io, """<line x1="$(round(cx; digits=2))" y1="$(round(y_high; digits=2))" x2="$(round(cx; digits=2))" y2="$(round(y_low; digits=2))" stroke="#222" stroke-width="1.3"/>""")
            println(io, """<line x1="$(round(cx - 5; digits=2))" y1="$(round(y_high; digits=2))" x2="$(round(cx + 5; digits=2))" y2="$(round(y_high; digits=2))" stroke="#222" stroke-width="1.3"/>""")
            println(io, """<line x1="$(round(cx - 5; digits=2))" y1="$(round(y_low; digits=2))" x2="$(round(cx + 5; digits=2))" y2="$(round(y_low; digits=2))" stroke="#222" stroke-width="1.3"/>""")
        end

        print(io, svg_footer())
    end
end

function write_block_jackknife_samples_svg(path)
    width = 980
    height = 430
    labels = [string(i) for i in 1:10]
    colors = fill("#6aaed6", 10)
    colors[1] = "#d9d9d9"
    colors[2:4] .= "#6aaed6"
    colors[5:7] .= "#9ccf77"
    colors[8:10] .= "#f0b45d"

    open(path, "w") do io
        print(io, svg_header(width, height))
        println(io, """<text class="title" x="490" y="32" text-anchor="middle">Block Jackknife: drop the initial remainder, then leave one block out</text>""")
        println(io, """<text class="label" x="490" y="62" text-anchor="middle">Example: N = 10, block size = 3, so the first 1 point is dropped.</text>""")

        x0 = 260
        box_width = 48
        gap = 10
        println(io, """<text class="tick" x="$(x0 + 0.5 * (box_width + gap))" y="91" text-anchor="middle">drop</text>""")
        println(io, """<text class="tick" x="$(x0 + 2.5 * (box_width + gap))" y="91" text-anchor="middle">block 1</text>""")
        println(io, """<text class="tick" x="$(x0 + 5.5 * (box_width + gap))" y="91" text-anchor="middle">block 2</text>""")
        println(io, """<text class="tick" x="$(x0 + 8.5 * (box_width + gap))" y="91" text-anchor="middle">block 3</text>""")

        draw_box_row(io, labels, Int[], 105; row_label="trimmed blocks", x0=x0, box_width=box_width, gap=gap, colors=colors, dropped=[1])
        draw_box_row(io, labels, [2, 3, 4], 170; row_label="sample 1", x0=x0, box_width=box_width, gap=gap, colors=colors, dropped=[1])
        draw_box_row(io, labels, [5, 6, 7], 235; row_label="sample 2", x0=x0, box_width=box_width, gap=gap, colors=colors, dropped=[1])
        draw_box_row(io, labels, [8, 9, 10], 300; row_label="sample 3", x0=x0, box_width=box_width, gap=gap, colors=colors, dropped=[1])

        println(io, """<text class="tick" x="490" y="392" text-anchor="middle">The central value and block samples use the data after dropping the initial remainder.</text>""")
        print(io, svg_footer())
    end
end

function block_mean_scan(measurements, chosen_block_size)
    nsamples = length(measurements.abs_magnetizations)
    candidate_blocks = unique(vcat(1:2:19, 20:5:95, 100:25:475, 500:100:2000, [chosen_block_size]))
    blocks = filter(b -> div(nsamples, b) >= 20, sort(candidate_blocks))
    scan = jk_block_scan(measurements.abs_magnetizations; blocks=blocks)
    return (block_sizes=scan.block_sizes, block_counts=scan.nblocks, values=scan.values, errors=scan.errors)
end

function write_block_mean_scan_svg(path, measurements, analysis)
    width = 860
    height = 520
    full_scan = block_mean_scan(measurements, analysis.block_size)
    keep = findall(>=(20), full_scan.block_sizes)
    order = sortperm(full_scan.block_counts[keep])
    block_counts = full_scan.block_counts[keep][order]
    values = full_scan.values[keep][order]
    errors = full_scan.errors[keep][order]
    block_sizes = full_scan.block_sizes[keep][order]
    # The horizontal axis is log10(number of blocks).
    logx = log10.(block_counts)
    xmin, xmax = log10(block_counts[1]) - 0.05, log10(block_counts[end]) + 0.05
    ymin, ymax = data_range(vcat(values .- errors, values .+ errors))

    chosen_index = findfirst(==(analysis.block_size), block_sizes)
    chosen_nblock = chosen_index === nothing ? missing : block_counts[chosen_index]
    chosen_value = chosen_index === nothing ? missing : values[chosen_index]
    chosen_error = chosen_index === nothing ? missing : errors[chosen_index]

    open(path, "w") do io
        print(io, svg_header(width, height))
        println(io, """<text class="title" x="430" y="32" text-anchor="middle">Block Jackknife estimate of |m| versus number of blocks</text>""")
        left, top, plot_width, plot_height = 95, 75, 700, 340
        draw_axes(io, left, top, plot_width, plot_height, xmin, xmax, ymin, ymax; xlabel="number of blocks (log scale)", ylabel="|m|", draw_x_tick_labels=false)
        for tick in (20, 50, 100, 200, 500, 1000, 2000)
            xmin <= log10(tick) <= xmax || continue
            tx = sx(log10(tick), xmin, xmax, left, plot_width)
            println(io, """<text class="tick" x="$(round(tx; digits=2))" y="$(top + plot_height + 18)" text-anchor="middle">$tick</text>""")
        end

        d = path_points(logx, values, xmin, xmax, ymin, ymax, left, top, plot_width, plot_height)
        println(io, """<path d="$d" fill="none" stroke="#20639b" stroke-width="2"/>""")
        for (nblock, value, error) in zip(block_counts, values, errors)
            cx = sx(log10(nblock), xmin, xmax, left, plot_width)
            cy = sy(value, ymin, ymax, top, plot_height)
            y_low = sy(value - error, ymin, ymax, top, plot_height)
            y_high = sy(value + error, ymin, ymax, top, plot_height)
            println(io, """<line x1="$(round(cx; digits=2))" y1="$(round(y_high; digits=2))" x2="$(round(cx; digits=2))" y2="$(round(y_low; digits=2))" stroke="#222" stroke-width="1.1"/>""")
            println(io, """<line x1="$(round(cx - 4; digits=2))" y1="$(round(y_high; digits=2))" x2="$(round(cx + 4; digits=2))" y2="$(round(y_high; digits=2))" stroke="#222" stroke-width="1.1"/>""")
            println(io, """<line x1="$(round(cx - 4; digits=2))" y1="$(round(y_low; digits=2))" x2="$(round(cx + 4; digits=2))" y2="$(round(y_low; digits=2))" stroke="#222" stroke-width="1.1"/>""")
            println(io, """<circle cx="$(round(cx; digits=2))" cy="$(round(cy; digits=2))" r="3.2" fill="#20639b"/>""")
        end

        if chosen_index !== nothing
            cx = sx(log10(chosen_nblock), xmin, xmax, left, plot_width)
            cy = sy(chosen_value, ymin, ymax, top, plot_height)
            println(io, """<line x1="$(round(cx; digits=2))" y1="$top" x2="$(round(cx; digits=2))" y2="$(top + plot_height)" stroke="#b23a48" stroke-dasharray="5,5"/>""")
            println(io, """<circle cx="$(round(cx; digits=2))" cy="$(round(cy; digits=2))" r="5" fill="#b23a48"/>""")
            println(io, """<text class="label" x="$(round(cx + 8; digits=2))" y="$(round(cy - 10; digits=2))">b = $(analysis.block_size), Nblock = $chosen_nblock</text>""")
        end

        println(io, """<text class="tick" x="430" y="505" text-anchor="middle">Each point is a central value; the vertical bars are block Jackknife errors.</text>""")
        print(io, svg_footer())
    end
end

function write_block_error_scan_svg(path, measurements, analysis)
    width = 860
    height = 520
    scan = block_mean_scan(measurements, analysis.block_size)
    base_error = jk_meanerror(measurements.abs_magnetizations)[2]
    ratios = scan.errors ./ base_error
    tau = analysis.autocorrelation.abs_magnetization.tau_int
    plateau = sqrt(2 * tau)
    xs = Float64.(scan.block_sizes)
    xmin, xmax = 0.0, maximum(xs)
    ymin, ymax = 0.0, 1.15 * max(maximum(ratios), plateau)

    open(path, "w") do io
        print(io, svg_header(width, height))
        println(io, """<text class="title" x="430" y="32" text-anchor="middle">Block Jackknife error of &lt;|m|&gt; versus block size</text>""")
        left, top, plot_width, plot_height = 95, 75, 700, 340
        draw_axes(io, left, top, plot_width, plot_height, xmin, xmax, ymin, ymax; xlabel="block size b (measurements)", ylabel="error(b) / error(b = 1)")

        py = sy(plateau, ymin, ymax, top, plot_height)
        println(io, """<line x1="$left" y1="$(round(py; digits=2))" x2="$(left + plot_width)" y2="$(round(py; digits=2))" stroke="#b23a48" stroke-dasharray="6,5"/>""")
        println(io, """<text class="label" x="$(left + plot_width - 8)" y="$(round(py - 8; digits=2))" text-anchor="end" fill="#b23a48">sqrt(2 tau_int) = $(round(plateau; digits=2))</text>""")

        d = path_points(xs, ratios, xmin, xmax, ymin, ymax, left, top, plot_width, plot_height)
        println(io, """<path d="$d" fill="none" stroke="#20639b" stroke-width="2"/>""")
        for (x, r) in zip(xs, ratios)
            println(io, """<circle cx="$(round(sx(x, xmin, xmax, left, plot_width); digits=2))" cy="$(round(sy(r, ymin, ymax, top, plot_height); digits=2))" r="3" fill="#20639b"/>""")
        end

        bx = sx(analysis.block_size, xmin, xmax, left, plot_width)
        println(io, """<line x1="$(round(bx; digits=2))" y1="$top" x2="$(round(bx; digits=2))" y2="$(top + plot_height)" stroke="#555" stroke-dasharray="3,4"/>""")
        println(io, """<text class="label" x="$(round(bx + 6; digits=2))" y="$(top + plot_height - 10)">chosen b = $(analysis.block_size)</text>""")

        println(io, """<text class="tick" x="430" y="505" text-anchor="middle">The ratio grows with b and levels off near sqrt(2 tau_int) once b is much larger than tau_int.</text>""")
        print(io, svg_footer())
    end
end

function write_error_comparison_svg(path, analysis)
    width = 860
    height = 500
    labels = ["energy", "|m|", "chi'", "C", "Binder", "chi(m)"]
    ordinary = [
        analysis.primary.energy[2],
        analysis.primary.abs_magnetization[2],
        analysis.secondary.susceptibility[2],
        analysis.secondary.specific_heat[2],
        analysis.secondary.binder_ratio[2],
        analysis.secondary.susceptibility_signed[2],
    ]
    block = [
        analysis.block.energy[2],
        analysis.block.abs_magnetization[2],
        analysis.block.susceptibility[2],
        analysis.block.specific_heat[2],
        analysis.block.binder_ratio[2],
        analysis.block.susceptibility_signed[2],
    ]
    ratios = block ./ ordinary
    ymin = 0.0
    ymax = 1.15 * maximum(vcat(ratios, [1.0]))

    open(path, "w") do io
        print(io, svg_header(width, height))
        println(io, """<text class="title" x="430" y="32" text-anchor="middle">Block / ordinary Jackknife error ratio</text>""")
        left, top, plot_width, plot_height = 95, 75, 700, 330
        draw_axes(io, left, top, plot_width, plot_height, 0.5, length(labels) + 0.5, ymin, ymax; xlabel="observable", ylabel="error ratio", draw_x_tick_labels=false)
        one_y = sy(1.0, ymin, ymax, top, plot_height)
        println(io, """<line x1="$left" y1="$(round(one_y; digits=2))" x2="$(left + plot_width)" y2="$(round(one_y; digits=2))" stroke="#555" stroke-dasharray="5,5"/>""")

        bar_width = plot_width / length(labels) * 0.48
        for (i, ratio) in enumerate(ratios)
            cx = sx(i, 0.5, length(labels) + 0.5, left, plot_width)
            x = cx - bar_width / 2
            y = sy(ratio, ymin, ymax, top, plot_height)
            bottom = sy(0.0, ymin, ymax, top, plot_height)
            println(io, """<rect x="$(round(x; digits=2))" y="$(round(y; digits=2))" width="$(round(bar_width; digits=2))" height="$(round(bottom - y; digits=2))" fill="#b23a48" opacity="0.78"/>""")
            println(io, """<text class="tick" x="$(round(cx; digits=2))" y="$(top + plot_height + 22)" text-anchor="middle">$(svg_escape(labels[i]))</text>""")
            println(io, """<text class="tick" x="$(round(cx; digits=2))" y="$(round(y - 8; digits=2))" text-anchor="middle">$(round(ratio; digits=2))</text>""")
        end

        print(io, svg_footer())
    end
end

function main()
    mkpath(FIGURE_DIR)
    mkpath(RESULT_DIR)

    measurements = Ising2DTutorial.run_ising2d()
    analysis = Ising2DTutorial.analyze_measurements(measurements)

    open(joinpath(RESULT_DIR, "ising2d_output.txt"), "w") do io
        Ising2DTutorial.write_summary(io, measurements, analysis)
    end

    write_time_series_svg(joinpath(FIGURE_DIR, "ising2d_timeseries.svg"), measurements)
    write_jackknife_samples_svg(joinpath(FIGURE_DIR, "jackknife_leave_one_out.svg"))
    write_autocorrelation_svg(joinpath(FIGURE_DIR, "ising2d_autocorrelation.svg"), analysis)
    write_block_jackknife_samples_svg(joinpath(FIGURE_DIR, "block_jackknife_blocks.svg"))
    write_block_mean_scan_svg(joinpath(FIGURE_DIR, "block_jackknife_mean_vs_nblocks.svg"), measurements, analysis)
    write_histogram_svg(joinpath(FIGURE_DIR, "ising2d_histogram_abs_m.svg"), analysis.histogram)
    write_error_comparison_svg(joinpath(FIGURE_DIR, "ising2d_error_ratio.svg"), analysis)
    write_block_error_scan_svg(joinpath(FIGURE_DIR, "ising2d_block_error_scan.svg"), measurements, analysis)

    println("Wrote tutorial artifacts to:")
    println("  ", RESULT_DIR)
    println("  ", FIGURE_DIR)
end

main()
