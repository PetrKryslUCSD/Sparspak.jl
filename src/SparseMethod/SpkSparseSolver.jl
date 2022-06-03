"""
This a the outer layer of software that contains a "solver"
#
The main role of this layer is to ensure that subroutines
are called in the proper order, and to collect timing
statistics for the major steps in solving systems of
equations.

Hence, the process of solving such sparse linear systems consists of a number of
steps: 

1. Reordering of the matrix A 

2. Symbolic factorization of the(reordered) matrix A and the creation of the
data structures for the factorization and forward and backward substitution 

3. Putting numerical values of
A into the data structures 

4. Numerical factorization of A 

5. Forward and backward substitution (triangular solution) 
"""
module SpkSparseSolver

using ..SpkProblem: Problem
using ..SpkSparseBase: SparseBase
import ..SpkSparseBase: findorder, symbolicfactor, inmatrix, factor, triangularsolve

mutable struct SparseSolver{IT, FT}
    slvr::SparseBase{IT, FT}
    n::IT
    ma::IT
    na::IT
    mc::IT
    nc::IT
    inmatrixdone::Bool
    orderdone::Bool
    symbolicdone::Bool
    factordone::Bool
    refinedone::Bool
    condestdone::Bool
end

function SparseSolver(p::Problem)
    ma = p.nrows
    na = p.ncols
    mc = 0
    nc = 0
    n = ma
    slvr = SparseBase(p)
    orderdone = false
    symbolicdone = false
    inmatrixdone = false
    factordone = false
    refinedone = false
    condestdone = false
    return SparseSolver(slvr, n, ma, na, mc, nc, inmatrixdone, orderdone, symbolicdone, factordone, refinedone, condestdone)
end

"""
    findorder(s::SparseSolver{IT}, orderfunction::F) where {IT, F}

Find reordering of the coefficient matrix.

"""
function findorder(s::SparseSolver{IT}, orderfunction::F) where {IT, F}
    if (s.orderdone)
        return true 
    end
    findorder(s.slvr, orderfunction)
    s.orderdone = true
    return true
end

"""
    findorder(s::SparseSolver{IT}) where {IT, F}

Find reordering of the coefficient matrix using the default method.
"""
function findorder(s::SparseSolver{IT}) where {IT, F}
    if (s.orderdone)
        return true 
    end
    findorder(s.slvr)
    s.orderdone = true
    return true
end

"""
findorderperm(s::SparseSolver{IT}, perm) where {IT}

Find reordering of the coefficient matrix using a given permutation.
"""
function findorderperm(s::SparseSolver{IT}, perm) where {IT}
    if (s.orderdone) 
        return true 
    end
    findorder(s.slvr, perm)
    s.orderdone = true
    return true
end

"""
    symbolicfactor(s::SparseSolver{IT})

Symbolic factorization of the(reordered) matrix A and the creation of the
data structures for the factorization and forward and backward substitution. 

"""
function symbolicfactor(s::SparseSolver{IT}) where {IT}
    if (s.symbolicdone) 
        return true
    end
    if ( ! s.orderdone)
        @error "$(@__FILE__): Sequence error. Ordering not done yet."
        return false
    end
    symbolicfactor(s.slvr)
    s.symbolicdone = true
    return true
end

"""
    inmatrix(s::SparseSolver{IT}, p::Problem{IT}) where {IT}


Put numerical values of the matrix stored in the problem into the data
structures of the solver.

"""
function inmatrix(s::SparseSolver{IT}, p::Problem{IT}) where {IT}
    if ( ! s.symbolicdone)
        @error "$(@__FILE__): Sequence error. Symbolic factor not done yet."
        return false
    end
    success = inmatrix(s.slvr, p)
    s.inmatrixdone = true
    return success
end

"""
    factor(s::SparseSolver{IT}) where {IT}

Numerical factorization of the coefficient matrix.

"""
function factor(s::SparseSolver{IT}) where {IT}
    if ( ! s.inmatrixdone)
        @error "$(@__FILE__): Sequence error. Matrix input not done yet."
        return false
    end
    factor(s.slvr)
    if (s.slvr.errflag == 0) 
        s.factordone = true
        return true
    else
        return false
    end
end

"""
    triangularsolve(s::SparseSolver{IT},  p::Problem{IT}) where {IT}

Forward and backward substitution (triangular solution).
"""
function triangularsolve(s::SparseSolver{IT},  p::Problem{IT}) where {IT}
    if ( ! s.factordone)
        @error "$(@__FILE__): Sequence error. Factorization not done yet."
        return false
    end

    temp = p.rhs[1:p.nrows]
    @assert length(temp) == s.n

    triangularsolve(s, temp)

    p.x .= temp

    s.refinedone = false
    return true
end

"""
    triangularsolve(s::SparseSolver{IT, FT}, solution::Vector{FT}) where {IT, FT}

Forward and backward substitution (triangular solution).

Variant where the right-hand side vector is passed in.
"""
function triangularsolve(s::SparseSolver{IT, FT}, solution::Vector{FT}) where {IT, FT}
    if ( ! s.factordone)
        @error "$(@__FILE__): Sequence error. Factorization not done yet."
        return false
    end
    triangularsolve(s.slvr, solution)
    s.refinedone = false
    return true
end

"""
    solve(s::SparseSolver{IT}, p::Problem{IT}) where {IT}

Execute all the steps of the solution process.

Given asymmetric matrix `A`, the steps are:

1. Reordering of the matrix `A`. 

2. Symbolic factorization of the(reordered) matrix `A` and the creation of the
data structures for the factorization and forward and backward substitution. 

3. Putting numerical values of `A` into the data structures. 

4. Numerical factorization of `A`. 

5. Forward and backward substitution (triangular solution).

The solution can be retrieved as `p.x`.
"""
function solve(s::SparseSolver{IT}, p::Problem{IT}) where {IT}
    findorder(s) || ErrorException("Finding Order.")
    symbolicfactor(s)
    inmatrix(s, p) || ErrorException("Matrix input.")
    factor(s)
    triangularsolve(s, p)
    return true
end

end