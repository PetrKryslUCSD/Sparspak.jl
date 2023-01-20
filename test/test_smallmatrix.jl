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
#
function simpletest(;n=4)
    A = sparse(Diagonal(ones(n)))
    A[2,1]=-0.1
    pr = SpkProblem.Problem(n,n)
    SpkProblem.insparse!(pr, A)
    s = SpkSparseSolver.SparseSolver(pr)
    @test SpkSparseSolver.findorder!(s)
end

function testproblem(;n=4)
    pr = SpkProblem.Problem(n, n)
    for i=1:n
        SpkProblem.inaij!(pr,i,i,1.0)
    end
    SpkProblem.inaij!(pr,2,1,-0.1)
    s = SpkSparseSolver.SparseSolver(pr)
    @test SpkSparseSolver.findorder!(s)
end

testproblem()
simpletest()
ttsparspak()
