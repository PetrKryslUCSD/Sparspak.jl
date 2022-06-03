[![Project Status: Active – The project is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Build status](https://github.com/PetrKryslUCSD/Sparspak.jl/workflows/CI/badge.svg)](https://github.com/PetrKryslUCSD/Sparspak.jl/actions)
[![Code Coverage](https://codecov.io/gh/PetrKryslUCSD/FinEtools.jl/branch/master/graph/badge.svg)](https://app.codecov.io/gh/PetrKryslUCSD/Sparspak.jl)
[![Latest documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://petrkryslucsd.github.io/Sparspak.jl/latest)
[![Codebase Graph](https://img.shields.io/badge/Codebase-graph-green.svg)](https://octo-repo-visualization.vercel.app/?repo=PetrKryslUCSD/Sparspak.jl)

# Sparspak.jl

Translation of the well-known sparse matrix software Sparspak (Waterloo Sparse Matrix Package), solving
large sparse systems of linear algebraic equations. Sparspak is composed of the
subroutines from the book "Computer Solution of Large Sparse Positive Definite
Systems" by Alan George and Joseph Liu. Originally written in Fortran 77, later
rewritten in Fortran 90. Here is the software translated into Julia.

## News

- 06/03/2022: The sparse LU solver has been now rewritten and tested.

## Simple usage

Here is a function to make up a random-coefficient (but diagonally dominant) sparse matrix and a right hand side vector.
```
function makerandomproblem(n)
    spm = sprand(n, n, 1/n)
    spm = -spm - spm' + 20 * LinearAlgebra.I
    
    p = Sparspak.SpkProblem.Problem(n, n)
    Sparspak.SpkProblem.insparse(p, spm);
    Sparspak.SpkProblem.infullrhs(p, 1:n);
    return p
end
```
The sparse linear algebraic equation problem can be solved as:
```
function _test()
    p = makerandomproblem(301)
    
    s = SparseSolver(p)
    solve(s, p)
    A = outsparse(p)
    x = A \ p.rhs
    @test norm(p.x - x) / norm(x) < 1.0e-6

    return true
end

_test()
```
For more details see the file `test/test_sparse_method.jl`, module `msprs012`.

## Reference

Alan George, Joseph Liu,
Computer Solution of Large Sparse Positive Definite Systems,
Prentice Hall, 1981,
ISBN: 0131652745,
LC: QA188.G46.

## Additional documents

Some design documents are in the folder `docs`: 
[SIAM paper](docs/Object_Oriented_interface_to_Sparspak.pdf), and the [User guide](docs/guide.pdf). These documents are only for
the Fortran version of the package.
