#using LinearSolve
using Test
using Sparspak
using Random, SparseArrays, LinearAlgebra
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkSparseSolver

# function ttlinsolve(;n=4,p=0.3)
#     for i=1:1000
#         println(i)
#         A = sprand(n, n, p) + I
#         b = rand(n)
#         prob = LinearProblem(A, b)
#         sol = solve(prob, SparspakFactorization())
#     end
# end

function ttsparspaklu(;n=4,p=0.3)
    for i=1:1000
        println(i)
        A = sprand(n, n, p) + I
        b = rand(n)
        display(Matrix(A))
        x=sparspaklu(A)\b
        @assert norm(x - A\b)<1.0e-10
    end
end

function ttsparspak(;n=4,p=0.3)
    for i=1:1000
        println(i)
        A = sprand(n, n, p) + I
        b = rand(n)
        @show A.colptr
        @show A.rowval
        display(Matrix(A))
        
        pr = SpkProblem.Problem(n,n)
        SpkProblem.insparse!(pr, A)
        Sparspak.SpkProblem.infullrhs!(pr, b)
        s = SpkSparseSolver.SparseSolver(pr)
        SpkSparseSolver.findorder!(s)
        SpkSparseSolver.symbolicfactor!(s)
        SpkSparseSolver.inmatrix!(s)
        SpkSparseSolver.factor!(s)
        SpkSparseSolver.solve!(s)

        @test norm(pr.x - A\b)<1.0e-10
    end
    true
end

#
# This is so far the simplest test problem occuring
# It fails in issymmetric, SpkGraph.jl:336
#
function simpletest1(;n=4)
    A = sparse(Diagonal(ones(n)))
    A[2,1]=-0.1
    pr = SpkProblem.Problem(n,n)
    SpkProblem.insparse!(pr, A)
    s = SpkSparseSolver.SparseSolver(pr)
    @test SpkSparseSolver.findorder!(s)
end

#
# May be another bug: fails in pkLUFactor.jl:245
#
function simpletest2()
    A=[1.21883    0.0  0.0      0.942235;
       0.0243952  1.0  0.0      0.0;
       0.0        0.0  1.53656  0.0;
       0.340938   0.0  0.0      1.0]

    A=sparse(A)
    pr = SpkProblem.Problem(4,4)
    SpkProblem.insparse!(pr, A)
    s = SpkSparseSolver.SparseSolver(pr)
    SpkSparseSolver.findorder!(s)
    SpkSparseSolver.symbolicfactor!(s)
    SpkSparseSolver.inmatrix!(s)
    @test SpkSparseSolver.factor!(s)
end

simpletest1()
simpletest2()
ttsparspak()
ttsparspaklu()

