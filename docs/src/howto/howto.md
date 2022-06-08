# How-to guide

## How to install


The latest release of Sparspak can be installed from the Julia REPL prompt with

```julia
julia> ]add Sparspak
```

The closing square bracket switches to the package manager interface and the `add`
commands installs Sparspak and any missing dependencies.  To return to the Julia
REPL hit the `delete` key.



## How to solve a random problem

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
    @show norm(p.x - x) / norm(x) 

    return true
end

_test()
```
For more details see the file `test/test_sparse_method.jl`, module `msprs016`.

## How to improve performance

Use `MKL`:

```
using LinearAlgebra
using SparseArrays
using MKL # <------------- Notice we put this before referencing Sparspak
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
    @show norm(p.x - x) / norm(x) 

    return true
end

_test()
```