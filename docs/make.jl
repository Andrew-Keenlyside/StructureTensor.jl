using Documenter
using StructureTensor

makedocs(
    sitename = "StructureTensor.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://Andrew-Keenlyside.github.io/StructureTensor.jl",
    ),
    modules = [StructureTensor],
    pages = [
        "Home" => "index.md",
        "User Guide" => "guide.md",
        "API Reference" => "api.md",
        "GPU Support" => "gpu.md",
        "File I/O" => "io.md",
        "Python Correspondence" => "python.md",
    ],
)

deploydocs(
    repo = "github.com/Andrew-Keenlyside/StructureTensor.jl.git",
)
