module m_simpletest
using Test
using Sparspak
using Random, SparseArrays, LinearAlgebra
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkSparseSolver

#
# This is so far the simplest test problem occuring
#
function simpletest(;n=4)
    A = sparse(Diagonal(ones(n)))
    A[2,1]=-0.1
    pr = SpkProblem.Problem(n,n, 10)
    @test SpkProblem.insparse!(pr, A)
    
    s = SpkSparseSolver.SparseSolver(pr)
    @test SpkSparseSolver.findorder!(s)
end

simpletest()
end

module m_simpletest1
using Test
using Sparspak
using Random, SparseArrays, LinearAlgebra
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkProblem: inaij!, inbi!, outsparse
using Sparspak.SpkSparseSolver
using Sparspak.SpkSparseSolver: SparseSolver, findorder!, symbolicfactor!, inmatrix!, factor!, solve!,  triangularsolve!

function test()
    p = SpkProblem.Problem(4, 4)
    inaij!(p, 1, 1, 1.21883);   
    inaij!(p, 2, 1, 0.0243952);
    inaij!(p, 4, 1, 0.340938);
    inaij!(p, 2, 2, 1.0);
    inaij!(p, 3, 3, 1.53656);
    inaij!(p, 1, 4, 0.942235);
    inaij!(p, 4, 4, 1.0);
    inbi!(p, 1, 1.0)
    A = outsparse(p)
    # @show   Matrix(A)
    # @show p.rhs
    x1 = A \ p.rhs
    s = SpkSparseSolver.SparseSolver(p)
    solve!(s)

    A = outsparse(p)
    x = A \ p.rhs
    @test norm(p.x - x) / norm(x) < 1.0e-6
    @test norm(p.x - x1) / norm(x1) < 1.0e-6
end

test()
end

module m_inconsistency_data_structure_symm
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkProblem: inaij!, inbi!, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, findorder!, symbolicfactor!, inmatrix!, factor!, solve!,  triangularsolve!

function makeaproblem()
    # Matrix is not symmetric, but it does have a symmetric structure.
    n = 4
    A = [1.21883    -0.02432  0.0     0.942235;
        0.0243952    1.0      0.0      0.0;
        0.0          0.0      1.53656  0.0;
        0.340938     0.0      0.0      1.0]

    p = SpkProblem.Problem(n, n)
    for i in 1:n, j in 1:n
        if A[i, j] != 0.0
            inaij!(p, i, j, A[i, j])
        end
    end
    inbi!(p, 1, 1.0)
    
    return p
end

function _test()
    p = makeaproblem()
    
    s = SparseSolver(p)
    
    solve!(s)

    A = outsparse(p)
    x = A \ p.rhs
    @test norm(p.x - x) / norm(x) < 1.0e-6

    return true
end

_test()
end # module

module m_inconsistency_data_structure_unsymm
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkProblem: inaij!, inbi!, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, findorder!, symbolicfactor!, inmatrix!, factor!, solve!,  triangularsolve!

function makeaproblem()
    # Matrix is not symmetric, and it does not have a symmetric structure.
    n = 4
    A = [1.21883    -0.02432  0.0      0.0;
        0.0          1.0      0.0      0.0;
        0.0          0.0      1.53656  0.0;
        0.340938     0.0      0.0      1.0]

    p = SpkProblem.Problem(n, n)
    for i in 1:n, j in 1:n
        if A[i, j] != 0.0
            inaij!(p, i, j, A[i, j])
        end
    end
    inbi!(p, 1, 1.0)
    
    return p
end

function _test()
    p = makeaproblem()
    
    s = SparseSolver(p)
    
    solve!(s)

    A = outsparse(p)
    x = A \ p.rhs
    @test norm(p.x - x) / norm(x) < 1.0e-6

    return true
end

_test()
end # module