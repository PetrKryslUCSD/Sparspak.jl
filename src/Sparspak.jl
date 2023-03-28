module Sparspak

include("Utilities/SpkUtilities.jl")
include("Utilities/GenericBlasLapackFragments.jl")
include("Grid/SpkGrid.jl")
include("Problem/SpkProblem.jl")
include("Ordering/SpkOrdering.jl")
include("Graph/SpkGraph.jl")
include("ETree/SpkETree.jl")
include("SparseSpdMethod/SpkMMD.jl")
include("SparseSpdMethod/SpkSymFct.jl")
include("SparseSpdMethod/SpkSpdMMOps.jl")
include("SparseSpdMethod/SpkLDLtFactor.jl")
include("SparseSpdMethod/SpkSparseSpdBase.jl")
include("SparseSpdMethod/SpkSparseSpdSolver.jl")
include("SparseMethod/SpkLUFactor.jl")
include("SparseMethod/SpkSparseBase.jl")
include("SparseMethod/SpkSparseSolver.jl")

using .SpkProblem
"""
    const Problem = SpkProblem

The module that defines the Sparspak problem.
"""
const Problem = SpkProblem

using .SpkSparseSolver
"""
    const SparseSolver = SpkSparseSolver

The module that defines a sparse-matrix LU solver.
"""
const SparseSolver = SpkSparseSolver


include("SparseCSCInterface/SparseCSCInterface.jl")
import .SparseCSCInterface: sparspaklu, sparspaklu!
export sparspaklu, sparspaklu!

end # module
