# Run every ```julia block of the READMEs, so the examples cannot go stale.
# Blocks with the package prompt (`pkg>`) or a plotting package are skipped.

function readme_julia_blocks(path)
    blocks = String[]
    for m in eachmatch(r"```julia\n(.*?)```"s, read(path, String))
        code = m.captures[1]
        occursin("pkg>", code) && continue
        occursin("using Plots", code) && continue
        push!(blocks, code)
    end
    return blocks
end

@testset "README examples: $(name)" for name in ("README.md", "README.ja.md")
    blocks = readme_julia_blocks(joinpath(@__DIR__, "..", name))
    @test length(blocks) >= 8

    mktempdir() do dir
        script = joinpath(dir, "readme_examples.jl")
        write(script, "using BlockJackknife\nusing Statistics\n\n" * join(blocks, "\n"))
        sandbox = Module(:ReadmeSandbox)
        @test (cd(() -> Base.include(sandbox, script), dir); true)
    end
end
