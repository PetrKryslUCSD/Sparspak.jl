# using Pkg;
# Pkg.activate()
# Pkg.instantiate()


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