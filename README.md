[![Project Status: Active – The project is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Build status](https://github.com/PetrKryslUCSD/Sparspak.jl/workflows/CI/badge.svg)](https://github.com/PetrKryslUCSD/Sparspak.jl/actions)
[![Code Coverage](https://codecov.io/gh/PetrKryslUCSD/FinEtools.jl/branch/master/graph/badge.svg)](https://app.codecov.io/gh/PetrKryslUCSD/Sparspak.jl)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://petrkryslucsd.github.io/Sparspak.jl/dev)
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

This code makes up a random-coefficient (but diagonally dominant) sparse matrix
and a simple right hand side vector. The sparse linear algebraic equation
problem is then solved with the LU factorization. The solution is tested
against the solution with the built-in solver.
```
using LinearAlgebra
using SparseArrays
using Sparspak.Problem: Problem, insparse!, outsparse, infullrhs!
using Sparspak.SparseSolver: SparseSolver, solve!

function _test()
    n = 1357
    A = sprand(n, n, 1/n)
    A = -A - A' + 20 * LinearAlgebra.I
    
    p = Problem(n, n)
    insparse!(p, A);
    infullrhs!(p, 1:n);
    
    s = SparseSolver(p)
    solve!(s, p)
    A = outsparse(p)
    x = A \ p.rhs
    @show norm(p.x - x) / norm(x) < 1.0e-6

    return true
end

_test()
```
For more details see the file `test/test_sparse_method.jl`, module `msprs016`.

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
