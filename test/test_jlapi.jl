using Test

module jl_mgrap001
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkGraph

function _test()
    # Matrix from Figure 3.1.3
    M, N = 6, 6
    I = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6]
    J = [1, 2, 6, 1, 2, 3, 4, 2, 3, 5, 2, 4, 3, 5, 6, 1, 5, 6]
    V = [1.0 for _ in I]
    spm = sparse(I, J, V, M, N)
    graph = SpkGraph.Graph(spm)
    @test graph.xadj == [1, 3, 6, 8, 9, 11, 13]
    @test graph.adj == [2, 6, 1, 3, 4, 2, 5, 2, 3, 6, 1, 5]
    return true
end

_test()
end # module

module jl_mgrap002
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkGraph

function _test()
    # Matrix from Figure 3.1.3
    M, N = 6, 6
    I = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6]
    J = [1, 2, 6, 1, 2, 3, 4, 2, 3, 5, 2, 4, 3, 5, 6, 1, 5, 6]
    V = [1.0 for _ in I]
    spm = sparse(I, J, V, M, N)
    graph = SpkGraph.Graph(spm)
    @test SpkGraph.isstructuresymmetric(graph)
    return true
end

_test()
end # module

module jl_mgrap003
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkGraph

function _test()
    # Matrix from Figure 3.1.3, with an element (5, 3) missing, hence unsymmetric.
    M, N = 6, 6
    I = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6]
    J = [1, 2, 6, 1, 2, 3, 4, 2, 3, 5, 2, 4, 5, 6, 1, 5, 6]
    V = [1.0 for _ in I]
    spm = sparse(I, J, V, M, N)
    graph = SpkGraph.Graph(spm)
    @test !SpkGraph.isstructuresymmetric(graph)
    return true
end

_test()
end # module


module jl_mgrap004
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkGraph

function _test()
    # Matrix from Figure 3.1.3, with an element (5, 3) missing, hence unsymmetric.
    M, N = 6, 6
    I = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6]
    J = [1, 2, 6, 1, 2, 3, 4, 2, 3, 5, 2, 4, 5, 6, 1, 5, 6]
    V = [1.0 for _ in I]
    spm = sparse(I, J, V, M, N)
    graph = SpkGraph.Graph(spm)
    SpkGraph.makestructuresymmetric(graph)
    @test SpkGraph.isstructuresymmetric(graph)
    @test graph.xadj == [1, 3, 6, 8, 9, 11, 13]
    @test graph.adj == [2, 6, 1, 3, 4, 2, 5, 2, 3, 6, 1, 5]
    return true
end

_test()
end # module

module jl_msolver001
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkSparseSolver: SparseSolver, findorder!, symbolicfactor!, inmatrix!, factor!, solve!,  triangularsolve!



function _test(T=Float64, n=20)
    spm = sprand(T, n, n, 1/n)
    spm = -spm - spm' + 40 * LinearAlgebra.I
    slv = SparseSolver(spm)
    exsol = ones(T,n)
    rhs = spm*exsol
    sol = solve!(slv,rhs)
    @test sol≈exsol
end


_test()
end
