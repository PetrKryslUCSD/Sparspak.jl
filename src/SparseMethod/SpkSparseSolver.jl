# This a the outer layer of software that contains a "solver"
# The main role of this layer is to ensure that subroutines are called in the
# proper order.

module SpkSparseSolver

using ..SpkProblem: Problem
using ..SpkSparseBase: _SparseBase
import ..SpkSparseBase: _findorder!, _symbolicfactor!, _inmatrix!, _factor!, _triangularsolve!
using SparseArrays

"""
    SparseSolver{IT, FT}

Type of LU general sparse solver.
"""
mutable struct SparseSolver{IT, FT}
    p::Union{Problem{IT, FT},SparseMatrixCSC{FT,IT}}
    slvr::_SparseBase{IT, FT}
    n::IT
    ma::IT
    na::IT
    mc::IT
    nc::IT
    _inmatrixdone::Bool
    _orderdone::Bool
    _symbolicdone::Bool
    _factordone::Bool
    _trisolvedone::Bool
    _refinedone::Bool
    _condestdone::Bool
end

"""
    SparseSolver(p::Problem)

Create a sparse solver from a problem.

The solver pulls all it needs from the problem.
"""
function SparseSolver(p::Problem)
    ma = p.nrows
    na = p.ncols
    mc = 0
    nc = 0
    n = ma
    slvr = _SparseBase(p)
    _orderdone = false
    _symbolicdone = false
    _inmatrixdone = false
    _factordone = false
    _trisolvedone = false
    _refinedone = false
    _condestdone = false
    return SparseSolver(p, slvr, n, ma, na, mc, nc, _inmatrixdone, _orderdone, _symbolicdone, _factordone, _trisolvedone, _refinedone, _condestdone)
end

"""
    solve!(s::SparseSolver{IT}) where {IT}

Execute all the steps of the solution process.

Given a symmetric matrix `A`, the steps are:

1. Reordering of the matrix `A`. 
2. Symbolic factorization of the(reordered) matrix `A`. 
3. Putting numerical values of `A` into the data structures. 
4. Numerical factorization of `A`. 
5. Forward and backward substitution (triangular solution).

The solution can be retrieved as `p.x`.
"""
function solve!(s::SparseSolver{IT}) where {IT}
    findorder!(s) || ErrorException("Finding Order.")
    symbolicfactor!(s) || ErrorException("Symbolic Factorization.")
    inmatrix!(s) || ErrorException("Matrix input.")
    factor!(s) || ErrorException("Numerical Factorization.")
    triangularsolve!(s) || ErrorException("Triangular Solve.")
    return true
end



"""
    findorder!(s::SparseSolver{IT}, orderfunction::F) where {IT, F}

Find reordering of the coefficient matrix.

- `orderfunction`: ordering function

If ordering has already been done for the solver, nothing happens. Otherwise,
the order function is applied.

Finding the ordering invalidates symbolic factorization.
"""
function findorder!(s::SparseSolver{IT}, orderfunction::F) where {IT, F}
    if (s._orderdone)
        return true 
    end
    _findorder!(s.slvr, orderfunction)
    s._orderdone = true
    s._symbolicdone = false
    return true
end

"""
    findorder!(s::SparseSolver{IT}) where {IT}

Find reordering of the coefficient matrix using the default method.

If ordering has already been done for the solver, nothing happens. Otherwise,
the order function is applied.

Finding the ordering invalidates symbolic factorization.
"""
function findorder!(s::SparseSolver{IT}) where {IT}
    if (s._orderdone)
        return true
    end
    _findorder!(s.slvr)
    s._orderdone = true
    s._symbolicdone = false
    return true
end

"""
    findorderperm!(s::SparseSolver{IT}, perm) where {IT}

Find reordering of the coefficient matrix using a given permutation.

If ordering has already been done for the solver, nothing happens. Otherwise,
the order function is applied.

Finding the ordering invalidates symbolic factorization.
"""
function findorderperm!(s::SparseSolver{IT}, perm) where {IT}
    if (s._orderdone) 
        return true 
    end
    _findorder!(s.slvr, perm)
    s._orderdone = true
    s._symbolicdone = false
    return true
end

"""
    symbolicfactor!(s::SparseSolver{IT})

Symbolic factorization of the(reordered) matrix A.

Create the data structures for the factorization and forward and backward
substitution. 

A symbolic factorization invalidates the input of the matrix.
"""
function symbolicfactor!(s::SparseSolver{IT}) where {IT}
    if (s._symbolicdone) 
        return true
    end
    if ( ! s._orderdone)
        error("Sequence error. Ordering not done yet.")
        return false
    end
    _symbolicfactor!(s.slvr)
    s._symbolicdone = true
    s._inmatrixdone = false
    return true
end

"""
    inmatrix!(s::SparseSolver{IT}) where {IT}

Put numerical values of the matrix stored in the problem into the data
structures of the solver.

If a matrix has been input before, and has not been invalidated by symbolic
factorization, nothing is done.

Input of the numerical values of the matrix invalidates the factorization.
"""
function inmatrix!(s::SparseSolver{IT}) where {IT}
    if (s._inmatrixdone) 
        return true
    end
    if ( ! s._symbolicdone)
        error("Sequence error. Symbolic factor not done yet.")
        return false
    end
    success = _inmatrix!(s.slvr, s.p)
    s._inmatrixdone = true
    s._factordone = false
    return success
end

"""
    factor!(s::SparseSolver{IT}) where {IT}

Numerical factorization of the coefficient matrix.

Numerical factorization invalidates the triangular solve.
"""
function factor!(s::SparseSolver{IT}) where {IT}
    @debug "factor! done? $(s._factordone)"
    if (s._factordone) 
        return true
    end
    if ( ! s._inmatrixdone)
        error("Sequence error. Matrix input not done yet.")
        return false
    end

    s._trisolvedone = false
    _factor!(s.slvr)
    if (s.slvr.errflag == 0) 
        s._factordone = true
        return true
    else
        return false
    end
end

"""
    triangularsolve!(s::SparseSolver{IT}) where {IT}

Forward and backward substitution (triangular solution).

The triangular solve is only done if it hasn't been done before; otherwise
nothing is done and the solution is the one obtained before.
"""
function triangularsolve!(s::SparseSolver{IT}) where {IT}
    if (s._trisolvedone) 
        return true
    end
    if ( ! s._factordone)
        error("Sequence error. Factorization not done yet.")
        return false
    end

    temp = deepcopy(s.p.rhs[1:s.p.nrows])
    @assert length(temp) == s.n

    triangularsolve!(s, temp)

    s.p.x .= temp

    s._trisolvedone = true
    s._refinedone = false
    return true
end

"""
    triangularsolve!(s::SparseSolver{IT, FT}, rhs::Vector{FT}) where {IT, FT}

Forward and backward substitution (triangular solution).

Variant where the right-hand side vector is passed in. This always triggers a
triangular solve.
"""
function triangularsolve!(s::SparseSolver{IT, FT}, rhs::AbstractVector{FT}) where {IT, FT}
    if ( ! s._factordone)
        error("Sequence error. Factorization not done yet.")
        return false
    end
    _triangularsolve!(s.slvr, rhs)
    s._trisolvedone = true
    s._refinedone = false
    return true
end

end
