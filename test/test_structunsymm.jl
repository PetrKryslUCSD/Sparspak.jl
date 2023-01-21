module structurally_unsymmetric
using Test
using Sparspak
using Random, SparseArrays, LinearAlgebra
using Sparspak.SpkOrdering
using Sparspak.SpkProblem
using Sparspak.SpkSparseSolver


#
# Used to fail in issymmetric, SpkGraph.jl:336
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
# Used to fail in SpkLUFactor.jl:245
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

#
# Used to fail in  SpkSparseBase.jl:390
#
function simpletest3()
    A=[   1.0       0.0       0.0      0.0;
          0.0       1.45565   0.0      0.0;
          0.0       0.0       1.11667  0.0;
          0.511585  0.159678  0.0      1.0]
    A=sparse(A)
    pr = SpkProblem.Problem(4,4)
    SpkProblem.insparse!(pr, A)
    s = SpkSparseSolver.SparseSolver(pr)
    SpkSparseSolver.findorder!(s)
    SpkSparseSolver.symbolicfactor!(s)
    SpkSparseSolver.inmatrix!(s)
    @test SpkSparseSolver.factor!(s)
end


function ttsparspak(;n=4,p=0.3)
    for i=1:1000
        A = sprand(n, n, p) + I
        b = rand(n)
        
        pr = SpkProblem.Problem(n,n)
        SpkProblem.insparse!(pr, A)
        Sparspak.SpkProblem.infullrhs!(pr, b)
        s = SpkSparseSolver.SparseSolver(pr)
        SpkSparseSolver.findorder!(s)
        SpkSparseSolver.symbolicfactor!(s)
        SpkSparseSolver.inmatrix!(s)
        SpkSparseSolver.factor!(s)
        SpkSparseSolver.solve!(s)
        @assert norm(pr.x - A\b)<1.0e-9
    end
    @test true
end


function ttsparspaklu(;n=4,p=0.3)
    for i=1:1000
        # println(i)
        A = sprand(n, n, p) + I
        b = rand(n)
        # display(Matrix(A))
        x=sparspaklu(A)\b
        @assert norm(x - A\b)<1.0e-9
    end
    @test true
end



simpletest1()
simpletest2()
simpletest3()
ttsparspak()
ttsparspaklu()

end
