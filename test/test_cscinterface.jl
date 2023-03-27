using Test

module csc_mgrap001
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

module csc_mgrap002
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

module csc_mgrap003
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


module csc_mgrap004
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

module csc_msolver001
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkSparseSolver: SparseSolver, solve!



function _test(T=Float64, n=20)
    spm = sprand(T, n, n, 1/n)
    spm = -spm - spm' + 40 * LinearAlgebra.I
    slv = SparseSolver(spm)
    exsol = ones(T,n)
    rhs = spm*exsol
    @test  solve!(slv,rhs)
    @test rhs ≈ exsol
end


_test()
end


module csc_msolver002
using Test
using LinearAlgebra
using SparseArrays
using Sparspak
using Random
using MultiFloats, ForwardDiff
Random.rand(rng::AbstractRNG, ::Random.SamplerType{ForwardDiff.Dual{T,V,N}}) where {T,V,N} = ForwardDiff.Dual{T,V,N}(rand(rng,T))



function _test(T=Float64, n=20)
    spm = sprand(T, n, n, 1/n)
    spm = -spm - spm' + 40 * LinearAlgebra.I

    
    exsol = ones(T,n)
    rhs = spm*exsol
    lu=sparspaklu(spm)
    sol=lu\rhs
    @test sol≈exsol

    spm.nzval.-=0.1
    rhs = spm*exsol
    lu=sparspaklu!(lu,spm)
    sol=lu\rhs
    @test sol≈exsol
    
end


_test(Float64)
_test(Float64x2)
_test(ForwardDiff.Dual{Float64,Float64,1})
end


module mcsc020
using Test
using Logging
using LinearAlgebra
using SparseArrays
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkProblem: inaij!, inbi!, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, findorder!, symbolicfactor!, inmatrix!, factor!, solve!,  triangularsolve!

function maketridiagproblem(n)
    p = SpkProblem.Problem(n, n)
    for i in 1:(n-1)
        inaij!(p, i + 1, i, -1.0)
        inaij!(p, i, i, 4.0)
        inaij!(p, i, i + 1, -1.0)
        inbi!(p, i, 2.0 * i)
    end
    inaij!(p, n, n, 4.0)
    inbi!(p, n, 3.0 * n + 1.0)

    return p
end

function _test()
    p = maketridiagproblem(11)

    s = SparseSolver(p)

    A = outsparse(p)
    rhs = copy(p.rhs)
    xr = A \ rhs

    rhs = copy(p.rhs)
    x = s \ rhs
    @test norm(x - xr) / norm(xr) < 1.0e-6

    rhs = copy(p.rhs)
    ldiv!(s, rhs)
    x = copy(rhs)
    @test norm(x - xr) / norm(xr) < 1.0e-6

    rhs = copy(p.rhs)
    x = copy(rhs)
    ldiv!(x, s, rhs)
    @test norm(x - xr) / norm(xr) < 1.0e-6

    return true
end

_test()
end # module


module mcsc021
using Test
using Logging
using LinearAlgebra
using SparseArrays
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkProblem: inaij!, inbi!, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, findorder!, symbolicfactor!, inmatrix!, factor!, solve!,  triangularsolve!

function _test()
    n = 41
    spm = sprand(Float64, n, n, 1/n)
    spm = -spm - spm' + 40 * LinearAlgebra.I
    prhs = collect(ones(n))

    s = SparseSolver(spm)

    A = copy(spm)
    rhs = copy(prhs)
    xr = A \ rhs

    rhs = copy(prhs)
    x = s \ rhs
    @test norm(x - xr) / norm(xr) < 1.0e-6

    rhs = copy(prhs)
    ldiv!(s, rhs)
    x = copy(rhs)
    @test norm(x - xr) / norm(xr) < 1.0e-6

    rhs = copy(prhs)
    x = copy(rhs)
    ldiv!(x, s, rhs)
    @test norm(x - xr) / norm(xr) < 1.0e-6

    return true
end

_test()
end # module


module mcsc022
using Test
using Logging
using LinearAlgebra
using SparseArrays
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkProblem: inaij!, inbi!, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, findorder!, symbolicfactor!, inmatrix!, factor!, solve!,  triangularsolve!

function _test()
    # @show "===================="
    n = 41
    spm = sprand(Float64, n, n, 1/n)
    spm = -spm - spm' + 40 * LinearAlgebra.I
    prhs = collect(ones(n))

    s = SparseSolver(spm)

    A = copy(spm)
    rhs = copy(prhs)
    xr = A \ rhs

    rhs = copy(prhs)
    x = s \ rhs
    @test norm(x - xr) / norm(xr) < 1.0e-6
    @test s._factordone

    rhs = copy(prhs)
    x = s \ rhs
    @test norm(x - xr) / norm(xr) < 1.0e-6

    return true
end

_test()
end # module
