module Sparspak

include("Utilities/SpkUtilities.jl")
include("Grid/SpkGrid.jl")
include("Problem/SpkProblem.jl")
include("Ordering/SpkOrdering.jl")
include("Graph/SpkGraph.jl")
include("ETree/SpkETree.jl")
include("SparseSpdMethod/SpkMMD.jl")
include("SparseSpdMethod/SpkSymFct.jl")
include("SparseSpdMethod/SpkSpdMMops.jl")
include("SparseMethod/SpkLUFactor.jl")
include("SparseMethod/SpkSparseBase.jl")
include("SparseMethod/SpkSparseSolver.jl")

end # module
