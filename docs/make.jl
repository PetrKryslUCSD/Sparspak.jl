# using Pkg;
# Pkg.activate()
# Pkg.instantiate()
import Pkg; Pkg.add("Documenter")
# import Pkg; Pkg.add("Sparspak")

using Documenter, Sparspak

makedocs(
	modules = [Sparspak],
	doctest = false, clean = true,
	format = Documenter.HTML(prettyurls = true),
	authors = "Petr Krysl",
	sitename = "Sparspak.jl",
	pages = Any[
	"Home" => "index.md",
	"Pages" => Any[
        "guide/guide.md",
		"man/reference.md",
        ]
	)

deploydocs(
    repo = "github.com/PetrKryslUCSD/Sparspak.jl.git",
)