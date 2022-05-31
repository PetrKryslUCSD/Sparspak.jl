using Test

module mgrid001
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkProblem
using Sparspak.SpkGrid

function _test()
    M, N = 6, 6
    g = SpkGrid.Grid(M, N)
    @test g.h == M
    @test g.k == N
    @test g.v == [1 2 3 4 5 6; 7 8 9 10 11 12; 13 14 15 16 17 18; 19 20 21 22 23 24; 25 26 27 28 29 30; 31 32 33 34 35 36] 
    return true
end

_test()
end # module
