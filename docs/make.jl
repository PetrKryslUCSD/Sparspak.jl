# using Pkg;
# Pkg.activate()
# Pkg.instantiate()

using Pkg
pkg"add https://github.com/PetrKryslUCSD/Sparspak.jl.git#main"

using Documenter, Sparspak

makedocs(
	modules = [Sparspak],
	doctest = false, clean = true,
	format = Documenter.HTML(prettyurls = false),
	authors = "Petr Krysl",
	sitename = "Sparspak.jl",
	pages = Any[
	"Home" => "index.md",
	"Guide" => "guide/guide.md",
	"Reference" => "man/reference.md",
        ]
	)

deploydocs(
    repo = "github.com/PetrKryslUCSD/Sparspak.jl.git",
)