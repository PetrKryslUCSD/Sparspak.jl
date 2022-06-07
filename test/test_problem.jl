using Test

module mprob001
using Test
using LinearAlgebra
using SparseArrays
using Sparspak.SpkProblem

function _test()
    M, N = 15, 15
    p = SpkProblem.Problem(M, N)
    spm = sprand(M, N, 0.2)
    spm = spm + spm'
    SpkProblem.insparse(p, spm)
    A = SpkProblem.outsparse(p)
    @test spm - A == sparse(Int64[], Int64[], Float64[], M, N)
    return true
end

_test()
end # module


module mprob002
using Test
using LinearAlgebra
using Sparspak.SpkUtilities

function _test()
    __extend = SpkUtilities.__extend
    a = [0.3814043930778628 0.07295808459382358 0.9234303156836231 0.2501555942396908 0.9901631548893776
        0.5435778459423668 0.018608588657332392 0.30803793111612243 0.7851377981322851 0.7642933432653911
        0.45649294345436575 0.27984784492761505 0.4280638037095217 0.9280066808672661 0.685472177669261]
    a = __extend(a, 2, 4)
    a1 = [0.3814043930778628 0.07295808459382358 0.9234303156836231 0.2501555942396908; 
        0.5435778459423668 0.018608588657332392 0.30803793111612243 0.7851377981322851]
    @test norm(a[1:2, 1:4] - a1) / norm(a) < 1.0e-9
    a1 = [0.3814043930778628 0.07295808459382358 0.9234303156836231 0.2501555942396908 0.0; 
    0.5435778459423668 0.018608588657332392 0.30803793111612243 0.7851377981322851 0.0; 
    0.0 0.0 0.0 0.0 0.0]
    a = __extend(a, 3, 5)
    @test norm(a - a1) / norm(a) < 1.0e-9
    a = __extend(a, 3, 5)
    @test norm(a - a1) / norm(a) < 1.0e-9
    a = __extend(a, 2, 2)
    a1 = [0.3814043930778628 0.07295808459382358 Inf Inf Inf; 0.5435778459423668 0.018608588657332392 Inf Inf Inf; Inf Inf Inf Inf Inf]
    a = __extend(a, 3, 5, SpkUtilities._BIGGY())
    @test a == a1
    return true
end

_test()
end # module