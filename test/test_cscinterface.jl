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
    solve!(slv,rhs)

    @test rhs≈exsol
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
    sparspaklu!(lu,spm)
    sol=lu\rhs
    @test sol≈exsol

    # create a matrix with different sparsity pattern
    spm2 = spm + sprand(T, n, n, 1/n)
    rhs = spm2*exsol

    # test fails attempting to reuse factorisation lu
    @test_throws ErrorException sparspaklu!(lu,spm2; allow_pattern_change=false)

    # test with default allow_pattern_change == true
    sparspaklu!(lu,spm2)
    sol=lu\rhs
    @test sol≈exsol
    
end


_test(Float64)
_test(Float64x2)
_test(ForwardDiff.Dual{Float64,Float64,1})


function _test_asymmetric(T=Float64, n=20)
    spm = sprand(T, n, n, 1/n)
    spm = -spm + 40 * LinearAlgebra.I

    
    exsol = ones(T,n)
    rhs = spm*exsol
    lu=sparspaklu(spm)
    sol=lu\rhs
    @test sol≈exsol

    spm.nzval.-=0.1
    rhs = spm*exsol
    sparspaklu!(lu,spm)
    sol=lu\rhs
    @test sol≈exsol

    # create a matrix with different sparsity pattern
    spm2 = spm + sprand(T, n, n, 1/n)
    rhs = spm2*exsol

    # test fails attempting to reuse factorisation lu
    @test_throws ErrorException sparspaklu!(lu,spm2; allow_pattern_change=false)

    # test with default allow_pattern_change=true (will redo symbolic factorization)
    sparspaklu!(lu,spm2)
    sol=lu\rhs
    @test sol≈exsol

    # test with uninitialized lu 
    lu_nofact = sparspaklu(spzeros(1, 1); factorize=false)
    sparspaklu!(lu_nofact,spm2; allow_pattern_change=false) # always allow update of unfactorized lu
    sol=lu_nofact\rhs
    @test sol≈exsol

end

_test_asymmetric(Float64)

end


# Int32-indexed CSC matrices.  Sparspak historically required the index integer
# type to equal `BlasInt` (Int64) so a `SparseMatrixCSC{Float64,Int32}` failed
# in the constructors of `Graph` / `_SparseBase` / `Ordering` / `SparseSolver`
# (mismatched scalar/array element types) and in the LU factorization path
# (BLAS-typed scalars expected throughout). Exercise both index types here so
# the matrix builds, factors, and solves without copying to Int64.
module csc_int32_solver
using Test
using LinearAlgebra
using SparseArrays
using Sparspak
using Random

function _test(T, IT, n=8)
    Random.seed!(42)
    spm_default = sprand(T, n, n, 0.4)
    spm_default = spm_default + n * LinearAlgebra.I
    spm = SparseMatrixCSC{T, IT}(spm_default)
    @test typeof(spm) === SparseMatrixCSC{T, IT}

    # Solver constructed without factorization (the path LinearSolve.jl uses
    # during init_cacheval — this must not throw even if a downstream BLAS
    # call would not yet support `IT`).
    lu_nofact = sparspaklu(spm; factorize=false)
    @test lu_nofact isa Sparspak.SpkSparseSolver.SparseSolver{IT, T}

    # Full factor-and-solve.
    exsol = ones(T, n)
    rhs = spm * exsol
    lu = sparspaklu(spm)
    sol = lu \ rhs
    @test sol ≈ exsol

    # Update values only (same pattern) and re-solve.
    spm.nzval .-= T(0.1)
    rhs = spm * exsol
    sparspaklu!(lu, spm)
    sol = lu \ rhs
    @test sol ≈ exsol
end

for IT in (Int32, Int64)
    for T in (Float64, Float32, ComplexF64, ComplexF32)
        _test(T, IT)
    end
end

end
