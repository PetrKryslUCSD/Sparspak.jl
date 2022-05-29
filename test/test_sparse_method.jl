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
    return true
end

_test()
end # module