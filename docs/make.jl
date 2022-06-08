# using Pkg;
# Pkg.activate()
# Pkg.instantiate()

# using Pkg
# pkg"add https://github.com/PetrKryslUCSD/Sparspak.jl.git#main"

using Documenter, Sparspak

makedocs(
	modules = [Sparspak],
	doctest = false, clean = true,
	format = Documenter.HTML(prettyurls = true, assets = ["assets/custom.css"], ),
	authors = "Petr Krysl",
	sitename = "Sparspak.jl",
	pages = Any[
	"Home" => "index.md",
	"How to" => "howto/howto.md",
	"Tutorials" => "tutorials/tutorials.md",
    "Concepts" => "concepts/concepts.md",
    "Reference" => ["man/types.md", "man/functions.md",],
        ]
	)

deploydocs(
    repo = "github.com/PetrKryslUCSD/Sparspak.jl.git",
)