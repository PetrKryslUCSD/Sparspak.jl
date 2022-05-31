using Test

module msprs001
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkProblem
using Sparspak.SpkSparseBase

function _test()
    # Matrix from Figure 3.1.3
    M, N = 6, 6
    I = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6]
    J = [1, 2, 6, 1, 2, 3, 4, 2, 3, 5, 2, 4, 3, 5, 6, 1, 5, 6]
    V = [1.0 for _ in I]
    spm = sparse(I, J, V, M, N)
    p = SpkProblem.Problem(M, N)
    SpkProblem.insparse(p, spm)
    s = SpkSparseBase.SparseBase(p)
    return true
end

_test()
end # module

module msprs002
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkProblem
using Sparspak.SpkSparseSolver

function _test()
    # Matrix from Figure 3.1.3
    M, N = 6, 6
    I = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6]
    J = [1, 2, 6, 1, 2, 3, 4, 2, 3, 5, 2, 4, 3, 5, 6, 1, 5, 6]
    V = [1.0 for _ in I]
    spm = sparse(I, J, V, M, N)
    p = SpkProblem.Problem(M, N)
    SpkProblem.insparse(p, spm)
    s = SpkSparseSolver.SparseSolver(p)
    return true
end

_test()
end # module

module msprs003
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkProblem: inaij, inbi, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, findorder

function maketridiagproblem(n)
    p = SpkProblem.Problem(n, n)
    for i in 1:(n-1)
        inaij(p, i + 1, i, -1.0)
        inaij(p, i, i, 4.0)
        inaij(p, i, i + 1, -1.0)
        inbi(p, i, 1.0)
    end
    inaij(p, n, n, 4.0)
    inbi(p, n, 1.0)
    return p
end

function _test()
    p = maketridiagproblem(11)
    # @show outsparse(p)
    s = SparseSolver(p)
    findorder(s)
    o =  s.slvr.order
    @test o.nrows == 11
    @test o.ncols == 11
    
    @test o.rperm == vec([11   1  10   2   9   3   8   4   7   5   6])
    @test o.rinvp == vec([2   4   6   8  10  11   9   7   5   3   1])
    @test o.cperm == vec([11   1  10   2   9   3   8   4   7   5   6])
    @test o.cinvp == vec([2   4   6   8  10  11   9   7   5   3   1])
    return true
end

_test()
end # module

module msprs004
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkProblem: inaij, inbi, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, findorder, symbolicfactor

function maketridiagproblem(n)
    p = SpkProblem.Problem(n, n)
    for i in 1:(n-1)
        inaij(p, i + 1, i, -1.0)
        inaij(p, i, i, 4.0)
        inaij(p, i, i + 1, -1.0)
        inbi(p, i, 1.0)
    end
    inaij(p, n, n, 4.0)
    inbi(p, n, 1.0)
    return p
end

function _test()
    p = maketridiagproblem(11)
    # @show outsparse(p)
    s = SparseSolver(p)
    findorder(s)
    symbolicfactor(s)

    @test s.slvr.xlnz == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]    
    @test s.slvr.xunz == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10] 
    
    @test s.slvr.xlindx == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    @test s.slvr.lindx == [1, 2, 2, 3, 3, 4, 4, 5, 5, 11, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11]
      
    return true
end

_test()
end # module


module msprs005
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkProblem: inaij, inbi, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, findorder, symbolicfactor, inmatrix

function maketridiagproblem(n)
    p = SpkProblem.Problem(n, n)
    for i in 1:(n-1)
        inaij(p, i + 1, i, -1.0)
        inaij(p, i, i, 4.0)
        inaij(p, i, i + 1, -1.0)
        inbi(p, i, 1.0)
    end
    inaij(p, n, n, 4.0)
    inbi(p, n, 1.0)
    return p
end

function _test()
    p = maketridiagproblem(11)
    # @show outsparse(p)
    s = SparseSolver(p)
    findorder(s)
    symbolicfactor(s)
    inmatrix(s, p)
    
   @test s.slvr.unz  == vec([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
   @test s.slvr.lnz  == vec([4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0, -1.0, -1.0,  4.0])
   @test s.slvr.xunz == vec([1           2           3           4        5           6           7           8           9          10          10            10])                       
     @test s.slvr.xlnz == vec([1           3           5           7           9          11          13          15          17          19          21          23])
    return true
end

_test()
end # module


module msprs006
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkProblem: inaij, inbi, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, findorder, symbolicfactor, inmatrix, factor

function maketridiagproblem(n)
    p = SpkProblem.Problem(n, n)
    for i in 1:(n-1)
        inaij(p, i + 1, i, -1.0)
        inaij(p, i, i, 4.0)
        inaij(p, i, i + 1, -1.0)
        inbi(p, i, 1.0)
    end
    inaij(p, n, n, 4.0)
    inbi(p, n, 1.0)
    return p
end

function _test()
    p = maketridiagproblem(11)
    # @show outsparse(p)
    s = SparseSolver(p)
    findorder(s)
    symbolicfactor(s)
    inmatrix(s, p)
    factor(s)
    s_xlnz = vec([ 1           3           5           7           9          11          13          15          17          19          21          23])                                 
    s_xunz = vec([ 1           2           3           4           5           6           7           8           9          10          10          10 ])
    @test s.slvr.xlnz == s_xlnz
    @test s.slvr.xunz == s_xunz
    s_lnz  = [  4.0, -0.25000000000000000, 3.7500000000000000, -0.26666666666666666, 3.7333333333333334, -0.26785714285714285, 3.7321428571428572 
, -0.26794258373205743, 3.7320574162679425, -0.26794871794871794, 4.0, -0.25000000000000000, 3.7500000000000000, -0.26666666666666666, 3.7333333333333334, -0.26785714285714285, 3.7321428571428572, -0.26794258373205743, 3.7320574162679425, -0.26794871794871794, -1.0000000000000000, 3.4641025641025642     ]
    s_unz = [ -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    @test norm(s.slvr.lnz - s_lnz) / norm(s_lnz) < 1e-6
    @test norm(s.slvr.unz - s_unz) / norm(s_unz) < 1e-6

    return true
end

_test()
end # module


module msprs007
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkProblem: inaij, inbi, outsparse
using Sparspak.SpkSparseSolver: SparseSolver, findorder, symbolicfactor, inmatrix, factor, solve

function maketridiagproblem(n)
    p = SpkProblem.Problem(n, n)
    for i in 1:(n-1)
        inaij(p, i + 1, i, -1.0)
        inaij(p, i, i, 4.0)
        inaij(p, i, i + 1, -1.0)
        inbi(p, i, 2.0 * i)
    end
    inaij(p, n, n, 4.0)
    inbi(p, n, 3.0 * n + 1.0)

    return p
end

function _test()
    p = maketridiagproblem(11)
    # @show outsparse(p)
    s = SparseSolver(p)
    findorder(s)
    symbolicfactor(s)
    inmatrix(s, p)
    factor(s)
    # before  LULSolve 
    # rhs = 34.0, 20.0, 18.0, 16.0, 14.0, 2.00, 4.00, 6.00, 8.00, 10.0, 12.0              
    # after  LULSolve  
    # rhs = 34.0, 28.5, 25.6, 22.857142857142858, 20.124401913875598, 2.00, 4.50, 7.2, 9.9285714285714288, 12.660287081339714, 20.784615384615385  
    solve(s, p)
    A = outsparse(p)
    @show x = A \ p.rhs
    @show p.x    

    return true
end

_test()
end # module
