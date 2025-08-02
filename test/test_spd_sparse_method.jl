using Test
using Logging: disable_logging, global_logger, ConsoleLogger, BelowMinLevel, min_enabled_level

# disable_logging(BelowMinLevel)
# global_logger(ConsoleLogger(stderr, BelowMinLevel))

# @show min_enabled_level(global_logger())

module mspdsprs001
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkProblem
using Sparspak.SpkSparseSpdBase

function _test()
    # Matrix from Figure 3.1.3
    M, N = 6, 6
    I = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6]
    J = [1, 2, 6, 1, 2, 3, 4, 2, 3, 5, 2, 4, 3, 5, 6, 1, 5, 6]
    V = [1.0 for _ in I]
    spm = sparse(I, J, V, M, N)
    p = SpkProblem.Problem(M, N)
    SpkProblem.insparse!(p, spm)
    s = SpkSparseSpdBase._SparseSpdBase(p)
    return true
end

_test()
end # module

module mspdsprs002
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkProblem
using Sparspak.SpkSparseSpdSolver: SpkSparseSpdSolver, findorder!, symbolicfactor!, inmatrix!

function _test()
    # Matrix from Figure 3.1.3
    M, N = 6, 6
    I = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6]
    J = [1, 2, 6, 1, 2, 3, 4, 2, 3, 5, 2, 4, 3, 5, 6, 1, 5, 6]
    V = [1.0 for _ in I]
    spm = sparse(I, J, V, M, N)
    p = SpkProblem.Problem(M, N)
    SpkProblem.insparse!(p, spm)
    s = SpkSparseSpdSolver.SparseSpdSolver(p)
    findorder!(s)
    symbolicfactor!(s)
    # inmatrix!(s)
    # @show s
    return true
end

_test()
end # module

module mspdsprs003
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkProblem
using Sparspak.SpkSparseSpdSolver: SpkSparseSpdSolver, findorder!, symbolicfactor!, inmatrix!, factor!

function _test()
    # Matrix from Figure 3.1.3
    M, N = 6, 6
    I = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6]
    J = [1, 2, 6, 1, 2, 3, 4, 2, 3, 5, 2, 4, 3, 5, 6, 1, 5, 6]
    V = [1.0 for _ in I]
    spm = sparse(I, J, V, M, N)
    p = SpkProblem.Problem(M, N)
    SpkProblem.insparse!(p, spm)
    s = SpkSparseSpdSolver.SparseSpdSolver(p)
    findorder!(s)
    symbolicfactor!(s)
    inmatrix!(s)
    factor!(s)
    return true
end

_test()
end # module


module mspdsprs004
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkProblem
using Sparspak.SpkProblem: outsparse
using Sparspak.SpkSparseSpdSolver: SpkSparseSpdSolver, findorder!, symbolicfactor!, inmatrix!, solve!

function _test()
    # Matrix from Figure 3.1.3
    M, N = 6, 6
    I = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6]
    J = [1, 2, 6, 1, 2, 3, 4, 2, 3, 5, 2, 4, 3, 5, 6, 1, 5, 6]
    V = [1.0 for _ in I]
    spm = sparse(I, J, V, M, N)
    p = SpkProblem.Problem(M, N)
    SpkProblem.insparse!(p, spm)
    s = SpkSparseSpdSolver.SparseSpdSolver(p)
    findorder!(s)
    symbolicfactor!(s)
    inmatrix!(s)
    solve!(s)
    
    A = Float64.(outsparse(p)) # no generic method for sparse...
    x = A \ p.rhs
    @test norm(p.x - x) / norm(x) < 1.0e-6
    return true
end

_test()
end # module

# module mspdsprs004
# using Test
# using LinearAlgebra
# using SparseArrays
# using Sparspak.SpkOrdering
# using Sparspak.SpkProblem
# using Sparspak.SpkProblem: inaij!, inbi!, outsparse
# using Sparspak.SpkSparseSolver: SparseSolver, findorder!, symbolicfactor!

# function maketridiagproblem(n)
#     p = SpkProblem.Problem(n, n)
#     for i in 1:(n-1)
#         inaij!(p, i + 1, i, -1.0)
#         inaij!(p, i, i, 4.0)
#         inaij!(p, i, i + 1, -1.0)
#         inbi!(p, i, 2.0 * i)
#     end
#     inaij!(p, n, n, 4.0)
#     inbi!(p, n, 3.0 * n + 1.0)

#     return p
# end

# function _test()
#     p = maketridiagproblem(11)
#     # @show outsparse(p)
#     s = SparseSolver(p)
#     findorder!(s)
#     symbolicfactor!(s)

#     @test s.slvr.xlnz == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
#     @test s.slvr.xunz == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10]
    
#     @test s.slvr.xlindx == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
#     @test s.slvr.lindx == [1, 2, 2, 3, 3, 4, 4, 5, 5, 11, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11]
      
#     return true
# end

# _test()
# end # module


# module mspdsprs005
# using Test
# using LinearAlgebra
# using SparseArrays
# using Sparspak.SpkOrdering
# using Sparspak.SpkProblem
# using Sparspak.SpkProblem: inaij!, inbi!, outsparse
# using Sparspak.SpkSparseSolver: SparseSolver, findorder!, symbolicfactor!, inmatrix!
# using MultiFloats

# function maketridiagproblem(n,T=Float64)
#     p = SpkProblem.Problem(n, n, 2*(n-1)+1, zero(T))
#     for i in 1:(n-1)
#         inaij!(p, i + 1, i, T(-1.0))
#         inaij!(p, i, i, T(4.0))
#         inaij!(p, i, i + 1, T(-1.0))
#         inbi!(p, i, T(2.0 * i))
#     end
#     inaij!(p, n, n, T(4.0))
#     inbi!(p, n, 3.0 * n + T(1.0))

#     return p
# end

# function _test(T=Float64)
#     p = maketridiagproblem(11,T)
#     # @show outsparse(p)
#     s = SparseSolver(p)
#     findorder!(s)
#     symbolicfactor!(s)
#     inmatrix!(s)
    
#    @test s.slvr.unz  == vec(T[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
#    @test s.slvr.lnz  == vec(T[4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0, -1.0, -1.0,  4.0])
#    @test s.slvr.xunz == vec([1           2           3           4        5           6           7           8           9          10          10            10])
#      @test s.slvr.xlnz == vec([1           3           5           7           9          11          13          15          17          19          21          23])
#     return true
# end

# _test()
# _test(Float64x1)
# _test(Float64x2)
# end # module


# module mspdsprs006
# using Test
# using LinearAlgebra
# using SparseArrays
# using Sparspak.SpkOrdering
# using Sparspak.SpkProblem
# using Sparspak.SpkProblem: inaij!, inbi!, outsparse
# using Sparspak.SpkSparseSolver: SparseSolver, findorder!, symbolicfactor!, inmatrix!, factor!

# using MultiFloats

# function maketridiagproblem(n,T=Float64)
#     p = SpkProblem.Problem(n, n, 2*(n-1)+1, zero(T))
#     for i in 1:(n-1)
#         inaij!(p, i + 1, i, T(-1.0))
#         inaij!(p, i, i, T(4.0))
#         inaij!(p, i, i + 1, T(-1.0))
#         inbi!(p, i, T(2.0 * i))
#     end
#     inaij!(p, n, n, T(4.0))
#     inbi!(p, n, 3.0 * n + T(1.0))

#     return p
# end


# function _test(T=Float64)
#     p = maketridiagproblem(11,T)
#     # @show outsparse(p)
#     s = SparseSolver(p)
#     findorder!(s)
#     symbolicfactor!(s)
#     inmatrix!(s)
#     factor!(s)
#     s_xlnz = vec([ 1           3           5           7           9          11          13          15          17          19          21          23])
#     s_xunz = vec([ 1           2           3           4           5           6           7           8           9          10          10          10 ])
#     @test s.slvr.xlnz == s_xlnz
#     @test s.slvr.xunz == s_xunz
#     s_lnz  = T[  4.0, -0.25000000000000000, 3.7500000000000000, -0.26666666666666666, 3.7333333333333334, -0.26785714285714285, 3.7321428571428572
# , -0.26794258373205743, 3.7320574162679425, -0.26794871794871794, 4.0, -0.25000000000000000, 3.7500000000000000, -0.26666666666666666, 3.7333333333333334, -0.26785714285714285, 3.7321428571428572, -0.26794258373205743, 3.7320574162679425, -0.26794871794871794, -1.0000000000000000, 3.4641025641025642     ]
#     s_unz = T[ -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
#     @test norm(s.slvr.lnz - s_lnz) / norm(s_lnz) < 1e-6
#     @test norm(s.slvr.unz - s_unz) / norm(s_unz) < 1e-6

#     return true
# end

# _test()
# _test(Float64x1)
# _test(Float64x2)
# end # module


# module mspdsprs007
# using Test
# using LinearAlgebra
# using SparseArrays
# using Sparspak.SpkOrdering
# using Sparspak.SpkProblem
# using Sparspak.SpkProblem: inaij!, inbi!, outsparse
# using Sparspak.SpkSparseSolver: SparseSolver, findorder!, symbolicfactor!, inmatrix!, factor!, solve!

# using MultiFloats

# function maketridiagproblem(n,T=Float64)
#     p = SpkProblem.Problem(n, n, 2*(n-1)+1, zero(T))
#     for i in 1:(n-1)
#         inaij!(p, i + 1, i, T(-1.0))
#         inaij!(p, i, i, T(4.0))
#         inaij!(p, i, i + 1, T(-1.0))
#         inbi!(p, i, T(2.0 * i))
#     end
#     inaij!(p, n, n, T(4.0))
#     inbi!(p, n, 3.0 * n + T(1.0))

#     return p
# end


# function _test(T=Float64)
#     p = maketridiagproblem(11,T)
    
#     s = SparseSolver(p)
#     findorder!(s)
#     symbolicfactor!(s)
#     inmatrix!(s)
#     factor!(s)
#     solve!(s)
#     A = Float64.(outsparse(p)) # no generic method for sparse...
#     x = A \ p.rhs
#     @test norm(p.x - x) / norm(x) < 1.0e-6

#     return true
# end

# _test()
# _test(Float64x1)
# _test(Float64x2)
# end # module


# module mspdsprs008
# using Test
# using LinearAlgebra
# using SparseArrays
# using Sparspak.SpkOrdering
# using Sparspak.SpkProblem
# using Sparspak.SpkProblem: inaij!, inbi!, outsparse
# using Sparspak.SpkSparseSolver: SparseSolver, findorder!, symbolicfactor!, inmatrix!, factor!, solve!

# using MultiFloats
# function maketridiagproblem(n,T=Float64)
#     p = SpkProblem.Problem(n, n, 2*(n-1)+1, zero(T))
#     for i in 1:(n-1)
#         inaij!(p, i + 1, i, T(-1.0))
#         inaij!(p, i, i, T(4.0))
#         inaij!(p, i, i + 1, T(-1.0))
#         inbi!(p, i, T(2.0 * i))
#     end
#     inaij!(p, n, n, T(4.0))
#     inbi!(p, n, 3.0 * n + T(1.0))

#     return p
# end

# function _test(T=Float64)
#     p = maketridiagproblem(11000,T)
    
#     s = SparseSolver(p)
#     findorder!(s)
#     symbolicfactor!(s)
#     inmatrix!(s)
#     factor!(s)
#     # before  LULSolve
#     # rhs = 34.0, 20.0, 18.0, 16.0, 14.0, 2.00, 4.00, 6.00, 8.00, 10.0, 12.0
#     # after  LULSolve
#     # rhs = 34.0, 28.5, 25.6, 22.857142857142858, 20.124401913875598, 2.00, 4.50, 7.2, 9.9285714285714288, 12.660287081339714, 20.784615384615385
#     solve!(s)
#     A = Float64.(outsparse(p))
#     x = A \ p.rhs
#     @test norm(p.x - x) / norm(x) < 1.0e-6

#     return true
# end

# _test()
# _test(Float64x1)
# _test(Float64x2)
# end # module


# module mspdsprs009
# using Test
# using LinearAlgebra
# using SparseArrays
# using Sparspak.SpkOrdering
# using Sparspak.SpkProblem
# using Sparspak.SpkProblem: inaij!, inbi!, outsparse
# using Sparspak.SpkSparseSolver: SparseSolver,  solve!

# using MultiFloats

# function maketridiagproblem(n,T=Float64)
#     p = SpkProblem.Problem(n, n, 2*(n-1)+1, zero(T))
#     for i in 1:(n-1)
#         inaij!(p, i + 1, i, T(-1.0))
#         inaij!(p, i, i, T(4.0))
#         inaij!(p, i, i + 1, T(-1.0))
#         inbi!(p, i, T(2.0 * i))
#     end
#     inaij!(p, n, n, T(4.0))
#     inbi!(p, n, 3.0 * n + T(1.0))

#     return p
# end

# function _test(T=Float64)
#     p = maketridiagproblem(1101)

#     s = SparseSolver(p)
#     solve!(s)
#     A = Float64.(outsparse(p))
#     x = A \ p.rhs
#     @test norm(p.x - x) / norm(x) < 1.0e-6

#     return true
# end

# _test()
# _test(Float64x1)
# _test(Float64x2)
# end # module


# module mspdsprs010
# using Test
# using LinearAlgebra
# using SparseArrays
# using Sparspak
# using Sparspak.SpkProblem: insparse!, outsparse, infullrhs!
# using Sparspak.SpkSparseSolver: SparseSolver, solve!

# using MultiFloats

# function makerandomproblem(n,T=Float64)
    
#     spm = sparse([1, 2, 10, 25, 27, 3, 18, 29, 4, 5, 27, 4, 5, 10, 6, 22, 7, 8,
#     20, 9, 16, 19, 2, 5, 10, 16, 22, 26, 11, 21, 12, 28, 13, 31, 14, 15, 14, 15, 9, 10, 16, 22, 17, 21, 23, 24, 3, 18, 23, 25, 9, 19, 8, 20, 11, 17, 21, 31, 6, 10, 16, 22, 23, 17, 18, 22, 23, 17, 24, 26, 27, 2, 18, 25, 28, 10, 24, 26, 2, 4, 24, 27, 12, 25, 28, 29, 3, 28, 29, 30, 29, 30, 13, 21, 31], [1, 2,
#     2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8, 8, 9, 9, 9, 10, 10, 10, 10,
#     10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 16, 16, 17, 17, 17,
#     17, 18, 18, 18, 18, 19, 19, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23,
#     23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 28,
#     28, 28, 28, 29, 29, 29, 29, 30, 30, 31, 31, 31], T[20.0, 20.0, -0.47048158392539896, -0.40373736633011315, -0.29904858543128643, 20.0, -0.8929708555328307, -0.25537381390842373, 20.0, -0.517204141985797, -0.4252150843692021, -0.517204141985797, 20.0, -0.007891864303108176, 20.0, -0.11187965399280864, 20.0, 20.0, -0.8431376173205444, 20.0, -0.867243409546438, -0.4387621276975887, -0.47048158392539896, -0.007891864303108176, 20.0, -0.5439984352593679, -0.7864685314415719, -0.6207822487609463, 20.0, -0.6015829471856056, 20.0, -0.958084621680583, 20.0, -0.21766891770662522, 20.0, -0.5616538599777784, -0.5616538599777784, 20.0, -0.867243409546438, -0.5439984352593679, 20.0, -0.48381531848154047, 20.0, -0.9600900246453297, -0.6481282913652262, -0.4940551331266394, -0.8929708555328307, 20.0, -0.6974069191376576, -0.9476414447186581, -0.4387621276975887, 20.0, -0.8431376173205444, 20.0, -0.6015829471856056,
#     -0.9600900246453297, 20.0, -0.9160515550545547, -0.11187965399280864, -0.7864685314415719, -0.48381531848154047, 20.0, -0.47566304294326156, -0.6481282913652262, -0.6974069191376576, -0.47566304294326156, 20.0, -0.4940551331266394, 20.0, -0.5220628570371777, -0.6929780200563616, -0.40373736633011315, -0.9476414447186581, 20.0, -0.013202153574752407, -0.6207822487609463, -0.5220628570371777, 20.0, -0.29904858543128643, -0.4252150843692021, -0.6929780200563616, 20.0, -0.958084621680583, -0.013202153574752407, 20.0, -0.14134601702703786, -0.25537381390842373, -0.14134601702703786, 20.0, -0.43870135287699785, -0.43870135287699785, 20.0, -0.21766891770662522, -0.9160515550545547,
#     20.0], 31, 31)
    
#     p = Sparspak.SpkProblem.Problem(n, n, nnz(spm))
#     Sparspak.SpkProblem.insparse!(p, spm);
#     Sparspak.SpkProblem.infullrhs!(p, 1:n)
#     return p
# end

# function _test(T=Float64)
#     p = makerandomproblem(31)
    
#     s = SparseSolver(p)
#     solve!(s)
    
#     A = Matrix(outsparse(p))
    
#     x = A \ p.rhs
#     @test norm(p.x - x) / norm(x) < 1.0e-6

#     return true
# end

# _test()
# _test(Float64x1)
# _test(Float64x2)
# end # module

# module mspdsprs011
# using Test
# using LinearAlgebra
# using SparseArrays
# using Sparspak
# using Sparspak.SpkProblem: insparse!, outsparse
# using Sparspak.SpkSparseSolver: SparseSolver
# using MultiFloats

# function _test(T=Float64)
#     n = 31
#     spm = sprand(T, n, n, 1/n)
#     spm = -spm - spm' + 20 * LinearAlgebra.I

#     p = Sparspak.SpkProblem.Problem(n, n, nnz(spm), zero(T))
#     Sparspak.SpkProblem.insparse!(p, spm);
#     A = outsparse(p)
#     @test A - spm == sparse(Int64[], Int64[], T[], 31, 31)


#     return true
# end

# _test()
# _test(Float64x1)
# _test(Float64x2)
# end # module


# module mspdsprs012
# using Test
# using LinearAlgebra
# using SparseArrays
# using Sparspak
# using Sparspak.SpkProblem: insparse!, outsparse
# using Sparspak.SpkSparseSolver: SparseSolver, solve!
# using MultiFloats

# function makerandomproblem(n,T=Float64)
#     spm = sprand(T, n, n, 1/n)
#     spm = -spm - spm' + n * LinearAlgebra.I
    
#     p = Sparspak.SpkProblem.Problem(n, n,nnz(spm), zero(T))
#     Sparspak.SpkProblem.insparse!(p, spm);
#     Sparspak.SpkProblem.infullrhs!(p, 1:n);
#     return p
# end

# function _test(T=Float64)

#     p = makerandomproblem(301,T)

#     s = SparseSolver(p)
#     solve!(s)
#     A = Matrix(outsparse(p))
#     x = A \ p.rhs
#     @test norm(p.x - x) / norm(x) < 1.0e-6

#     return true
# end

# _test()
# _test(Float64x1)
# _test(Float64x2)
# end # module


# # module mspdsprs013
# # using Test
# # using LinearAlgebra
# # using SparseArrays
# # using DataDrop
# # using Sparspak
# # using Sparspak.Problem: Problem, insparse, outsparse
# # using Sparspak.SparseSolver: SparseSolver, findorder, symbolicfactor, inmatrix, factor, solve
# # # using ProfileView
# # # using InteractiveUtils

# # function _test()
# #     K = DataDrop.retrieve_matrix("K63070.h5")
# #     # K = DataDrop.retrieve_matrix("K28782.h5")
# #     @show size(K)
# #     I, J, V = findnz(K)

# #     p = Problem(size(K)...)
# #     insparse(p, I, J, V);
# #     s = Sparspak.SpkSparseSolver.SparseSolver(p);
# #     @time findorder(s)
# #     symbolicfactor(s)
# #     inmatrix(s, p)
# #     @time factor(s)
# #     # @time factor(s)
# #     # @profview factor(s)
# #     @time solve(s, p);

# #     @info "$(@__FILE__): Now lu"
# #     @time lu(K)

# #     @info "$(@__FILE__): Now cholesky"
# #     @time cholesky(K)

# #     return true
# # end

# # _test()
# # end # module


# # module mspdsprs014
# # using Test
# # using LinearAlgebra
# # using SparseArrays
# # using Sparspak
# # using Sparspak.SpkProblem: insparse, outsparse
# # using Sparspak.SpkSparseSolver: SparseSolver, findorder, symbolicfactor, inmatrix, factor, solve
# # using Printf

# # function makerandomproblem(n)
# #     p = Sparspak.SpkProblem.Problem(n, n)
# #     spm = sprand(n, n, 1/n)
# #     spm = -spm - spm' + 20 * LinearAlgebra.I
    
# #     Sparspak.SpkProblem.insparse(p, spm);
# #     Sparspak.SpkProblem.infullrhs(p, 1:n);

# #     matrix = "matrix$(n).txt"
# #     I, J, V = findnz(spm)
# #     @show

# #     f = open(matrix, "w")
# #     @printf(f, "%d %d\n", size(spm, 1), size(spm, 2))
# #     @printf(f, "%d\n", length(I))
# #     for i in eachindex(I)
# #         @printf(f, "%d\n", I[i])
# #     end
# #     for i in eachindex(J)
# #         @printf(f, "%d\n", J[i])
# #     end
# #     for i in eachindex(V)
# #         @printf(f, "%f\n", V[i])
# #     end
# #     close(f)

# #     return p
# # end

# # function _test()
# #     p = makerandomproblem(11)
    
# #     s = SparseSolver(p)
# #     solve(s, p)
# #     A = outsparse(p)
# #     x = A \ p.rhs
# #     @test norm(p.x - x) / norm(x) < 1.0e-6

# #     return true
# # end

# # _test()
# # end # module


# module mspdsprs015
# using Test
# using LinearAlgebra
# using SparseArrays
# using Sparspak
# using Sparspak.Problem: Problem, insparse!, outsparse, infullrhs!
# using Sparspak.SparseSolver: SparseSolver, solve!

# using MultiFloats
# function _test(T=Float64)
#     n = 357
#     A = sprand(T, n, n, 1/n)
#     A = -A - A' + 20 * LinearAlgebra.I
    
#     p = Problem(n, n, nnz(A), zero(T))
#     insparse!(p, A);
#     infullrhs!(p, 1:n);
    
#     s = SparseSolver(p)
#     solve!(s)
#     A = Float64.(outsparse(p))
#     x = A \ p.rhs
#     @test norm(p.x - x) / norm(x) < 1.0e-6

#     return true
# end

# _test()
# _test(Float64x1)
# _test(Float64x2)
# end # module

# module mspdsprs016
# using Test
# using LinearAlgebra
# using SparseArrays
# using Sparspak.Problem: Problem, insparse!, outsparse, infullrhs!
# using Sparspak.SparseSolver: SparseSolver, solve!

# using MultiFloats
# function _test(T=Float64)
#     n = 1357
#     A = sprand(T, n, n, 1/n)
#     A = -A - A' + 20 * LinearAlgebra.I

#     p = Problem(n, n, nnz(A), zero(T))
#     insparse!(p, A);
#     infullrhs!(p, 1:n);

#     s = SparseSolver(p)
#     solve!(s)
#     A = Float64.(outsparse(p))
#     x = A \ p.rhs
#     @test norm(p.x - x) / norm(x) < 1.0e-6

#     return true
# end

# _test()
# _test(Float64x1)
# _test(Float64x2)
# end # module

# module mspdsprs017
# using Test
# using LinearAlgebra
# using SparseArrays
# using Sparspak
# using Sparspak.SpkProblem: outsparse, makegridproblem, infullrhs!
# using Sparspak.SpkSparseSolver: SparseSolver, solve!
# using Printf

# function _test()
#     p = makegridproblem(5, 3)
#     infullrhs!(p, 1:p.nrows);
    
#     s = SparseSolver(p)
#     A = outsparse(p)
#     solve!(s)
#     x = A \ p.rhs
#     @test norm(p.x - x) / norm(x) < 1.0e-6

#     return true
# end

# _test()
# end # module

# module mspdsprs018
# using Test
# using LinearAlgebra
# using SparseArrays
# using Sparspak
# using Sparspak.SpkProblem: outsparse, makegridproblem, makerhs!
# using Sparspak.SpkSparseSolver: SparseSolver, solve!
# using Printf

# function _test()
#     p = makegridproblem(5, 3)
#     makerhs!(p, Float64[]);
    
#     s = SparseSolver(p)
#     A = outsparse(p)
#     solve!(s)
#     x = A \ p.rhs
#     @test norm(p.x - x) / norm(x) < 1.0e-6

#     return true
# end

# _test()
# end # module

# module mspdsprs019
# using Test
# using LinearAlgebra
# using SparseArrays
# using Sparspak.SpkOrdering
# using Sparspak.SpkProblem
# using Sparspak.SpkProblem: inaij!, inbi!, outsparse
# using Sparspak.SpkSparseSolver: SparseSolver, findorder!, symbolicfactor!, inmatrix!, factor!, solve!,  triangularsolve!

# function maketridiagproblem(n)
#     p = SpkProblem.Problem(n, n)
#     for i in 1:(n-1)
#         inaij!(p, i + 1, i, -1.0)
#         inaij!(p, i, i, 4.0)
#         inaij!(p, i, i + 1, -1.0)
#         inbi!(p, i, 2.0 * i)
#     end
#     inaij!(p, n, n, 4.0)
#     inbi!(p, n, 3.0 * n + 1.0)

#     return p
# end

# function _test()
#     p = maketridiagproblem(11)
    
#     s = SparseSolver(p)
#     # The actions are purposefully mixed up
#     @test_throws ErrorException symbolicfactor!(s) == false # ordering not done
    
    
#     findorder!(s)
#     @test_throws ErrorException inmatrix!(s) == false

#     findorder!(s)
#     symbolicfactor!(s)
#     @test_throws ErrorException factor!(s) == false

#     findorder!(s)
#     symbolicfactor!(s)
#     inmatrix!(s)
#     @test_throws ErrorException triangularsolve!(s) == false
    
#     solve!(s)

#     A = outsparse(p)
#     x = A \ p.rhs
#     @test norm(p.x - x) / norm(x) < 1.0e-6

#     return true
# end

# _test()
# end # module

