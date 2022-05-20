using Test

module mutil001
using Test
using LinearAlgebra
using Sparspak.SpkUtilities

function _test()
    extend = SpkUtilities.extend
    a = [ 9.91099e-01
    8.26992e-01
    4.78277e-02
    7.21746e-01
    2.77298e-01
    8.00130e-01
    8.07221e-01]
    b = deepcopy(a)
    a = extend(a, 3)
    @test norm(a - b[1:3]) / norm(a) < 1.0e-9
    a = extend(a, 5, -1.0)
    c = vcat(b[1:3], [-1.0, -1.0])
    @test norm(a - c) / norm(a) < 1.0e-9
    a = deepcopy(b)
    a = extend(a, 5, -1.0)
    @test norm(a - b[1:5]) / norm(a) < 1.0e-9
    return true
end

_test()
end # module