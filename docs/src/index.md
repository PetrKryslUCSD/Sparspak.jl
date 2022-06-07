# Sparspak Documentation


## Package features

- Solves systems of coupled linear algebraic equations with a sparse coefficient matrix.
- Reorderings of various kinds are supported, including the Multiple Minimum Degree (MMD).
- Factorizations of various kinds are supported.
- Solutions with multiple right hand sides, and solutions with preserved structure but changed matrix coefficients are supported. 

## Installation

The latest release of Sparspak can be installed from the Julia REPL prompt with

```julia
julia> ]add Sparspak
```

The closing square bracket switches to the package manager interface and the `add`
commands installs Sparspak and any missing dependencies.  To return to the Julia
REPL hit the `delete` key.

## Simple example


This code makes up a random-coefficient (but diagonally dominant) sparse matrix
and a simple right hand side vector. The sparse linear algebraic equation
problem is then solved with the LU factorization. The solution is tested
against the solution with the built-in solver.
```
using Sparspak.Problem: Problem, insparse, outsparse, infullrhs
using Sparspak.SparseSolver: SparseSolver, solve

function _test()
    n = 1357
    A = sprand(n, n, 1/n)
    A = -A - A' + 20 * LinearAlgebra.I
    
    p = Problem(n, n)
    insparse(p, A);
    infullrhs(p, 1:n);
    
    s = SparseSolver(p)
    solve(s, p)
    A = outsparse(p)
    x = A \ p.rhs
    @test norm(p.x - x) / norm(x) < 1.0e-6

    return true
end

_test()
```
For more details see the file `test/test_sparse_method.jl`, module `msprs016`.


## User guide

```@contents
Pages = [
    "guide/guide.md",
]
Depth = 1
```

## Manual

The description of the types and the functions, organized by module and/or other logical principle.

```@contents
Pages = [
    "man/reference.md",
]
Depth = 3
```
