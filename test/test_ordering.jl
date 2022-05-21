using Test

module morde001
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkOrdering

function _test()
    M, N = 15, 15
    order = SpkOrdering.Ordering(M, N)
    @test order.rperm == collect(1:M)
    order = SpkOrdering.Ordering(M)
    @test order.rperm == collect(1:M)
    return true
end

_test()
end # module

