#
# """
# This a the outer layer of software that contains a "solver"
##
# The main role of this layer is to ensure that subroutines
# are called in the proper order, and to collect timing
# statistics for the major steps in solving systems of
# equations.
#
#
# """
module SpkSparseSpdSolver

using ..SpkProblem: Problem
using ..SpkSparseSpdBase: _SparseSpdBase
import ..SpkSparseSpdBase: _triangularsolve!
import ..SpkSparseSpdBase: _findorder!, _symbolicfactor!, _inmatrix!, _factor!
using SparseArrays


mutable struct SparseSpdSolver{IT, FT}
    p::Union{Problem{IT, FT},SparseMatrixCSC{FT,IT}}
    slvr::_SparseSpdBase{IT, FT}
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
"""
function SparseSpdSolver(p::Problem)
    ma = p.nrows
    na = p.ncols
    mc = 0
    nc = 0
    n = ma
    slvr = _SparseSpdBase(p)
    _orderdone = false
    _symbolicdone = false
    _inmatrixdone = false
    _factordone = false
    _trisolvedone = false
    _refinedone = false
    _condestdone = false
    return SparseSpdSolver(p, slvr, n, ma, na, mc, nc, _inmatrixdone, _orderdone, _symbolicdone, _factordone, _trisolvedone, _refinedone, _condestdone)
end

"""
    findorder!(s::SparseSpdSolver{IT}) where {IT}

Find reordering of the coefficient matrix using the default method.

If ordering has already been done for the solver, nothing happens. Otherwise,
the order function is applied.

Finding the ordering invalidates symbolic factorization.
"""
function findorder!(s::SparseSpdSolver{IT}, orderfunction::F) where {IT, F}
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
function findorder!(s::SparseSpdSolver{IT}) where {IT}
    if (s._orderdone)
        return true
    end
    _findorder!(s.slvr)
    s._orderdone = true
    s._symbolicdone = false
    return true
end

"""
    findorderperm!(s::SparseSpdSolver{IT}, perm) where {IT}

Find reordering of the coefficient matrix using a given permutation.

If ordering has already been done for the solver, nothing happens. Otherwise,
the order function is applied.

Finding the ordering invalidates symbolic factorization.
"""
function findorderperm!(s::SparseSpdSolver{IT}, perm) where {IT}
    if (s._orderdone)
        return true
    end
    _findorder!(s.slvr, perm)
    s._orderdone = true
    s._symbolicdone = false
    return true
end


"""
    symbolicfactor!(s::SparseSpdSolver{IT})

Symbolic factorization of the(reordered) matrix A.

Create the data structures for the factorization and forward and backward
substitution.

A symbolic factorization invalidates the input of the matrix.
"""
function symbolicfactor!(s::SparseSpdSolver{IT}) where {IT}
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
    inmatrix!(s::SparseSpdSolver{IT}) where {IT}

Put numerical values of the matrix stored in the problem into the data
structures of the solver.

If a matrix has been input before, and has not been invalidated by symbolic
factorization, nothing is done.

Input of the numerical values of the matrix invalidates the factorization.
"""
function inmatrix!(s::SparseSpdSolver{IT}) where {IT}
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
    factor!(s::SparseSpdSolver{IT}) where {IT}

Numerical factorization of the coefficient matrix.

Numerical factorization invalidates the triangular solve.
"""
function factor!(s::SparseSpdSolver{IT}) where {IT}
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
    solve!(s::SparseSpdSolver{IT}) where {IT}

Execute all the steps of the solution process.

Given a symmetric matrix `A`, the steps are:

1. Reordering of the matrix `A`.
2. Symbolic factorization of the(reordered) matrix `A`.
3. Putting numerical values of `A` into the data structures.
4. Numerical factorization of `A`.
5. Forward and backward substitution (triangular solution).

The solution can be retrieved as `p.x`.
"""
function solve!(s::SparseSpdSolver{IT}) where {IT}
    findorder!(s) || error("Finding Order.")
    symbolicfactor!(s) || error("Symbolic Factorization.")
    inmatrix!(s) || error("Matrix input.")
    factor!(s) || error("Numerical Factorization.")
    triangularsolve!(s) || error("Triangular Solve.")
    return true
end


"""
    triangularsolve!(s::SparseSpdSolver{IT}) where {IT}

Forward and backward substitution (triangular solution).

The triangular solve is only done if it hasn't been done before; otherwise
nothing is done and the solution is the one obtained before.
"""
function triangularsolve!(s::SparseSpdSolver{IT}) where {IT}
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
    triangularsolve!(s::SparseSpdSolver{IT, FT}, rhs::Vector{FT}) where {IT, FT}

Forward and backward substitution (triangular solution).

Variant where the right-hand side vector is passed in. This always triggers a
triangular solve.
"""
function triangularsolve!(s::SparseSpdSolver{IT, FT}, rhs::AbstractVector{FT}) where {IT, FT}
    if ( ! s._factordone)
        error("Sequence error. Factorization not done yet.")
        return false
    end
    _triangularsolve!(s.slvr, rhs)
    s._trisolvedone = true
    s._refinedone = false
    return true
end

# function sparsespdrefine(s, p, mtype, iterations)
#         type (SparseSpdSolver) :: s
#         type (problem) :: p
#         character (len = *), optional :: mtype
#         real (double) :: res(s.n)
#         real (double) :: s1, s2, ratio
#         integer, optional :: iterations
#         integer :: numiter, i, j
#         character (len = *), parameter :: fname = "sparsespdrefine:"

#         if (s.n == 0)
#             warning(fname, "this solver object is derived from ")
#             warning("an empty problem. no refinement done.")
#             return
#         end

#         if ( ! s.factordone)
#             warning(fname, "matrix has not been factored.")
#             warning("iterative refinement not done.")
#             return
#         end

#         gettime(s.refinetime)
#         inform("")
#         inform("begin refine ...")

#         res(1:s.n) = 0.0d0;


#         if (present(iterations))
#             numiter = iterations
#         else
#             numiter = 5
#         end

# #       Do iterative refinement.
#         for i = 1: numiter

# #           Find the residual.
#             computeresidual(p, res, mtype = mtype)

# #           Solve for the error.
#             triangularsolve(s, res)

# #           Adjust the solution.
#             p.x(1:s.n) = p.x(1:s.n) + res(1:s.n)

# #           Sum the error components.
#             s1 = 0.0d0
#             for j = 1: s.n
#                 s1 = s1 + abs(res(j))
#             end

# #           Sum the solution components.
#             s2 = 0.0d0
#             for j = 1: s.n
#                s2 = s2 + abs(p.x(j))
#             end

#             ratio = s1 / s2
#             FIXME write(stdout, *) "ratio:", ratio
#             FIXME write(stdout, "(1x, 4f19.13)") p.x(1:4)

#             if (logunit > 0)
#                 FIXME write(logunit, *) "ratio:", ratio
#                 FIXME write(logunit, "(1x, 4f19.13)") p.x(1:4)
#             end

#             if (ratio < 1d - 13) exit
#         end


#         gettime(s.refinetime, s.refinetime)
#         s.refinedone = true
#         inform("end refine")

#     end


# function sparsespdainverseonenorm(s, est)
#         type (SparseSpdSolver) :: s
#         real(double) :: est, altsgn, estold, temp, onenorma, x(s.n)
#         integer :: n, i, iter, j, jlast, isgn(s.n)
#         integer, parameter :: itmax = 5
#         real(double), parameter :: zero = 0.0d0, one = 1.0d0, two = 2.0d0
#         character (len = *), parameter :: fname = "sparsespdainverseonenorm:"

#         trace (fname, "problem object name is", s.objectname)

#         est = 0.0d0;  n = s.n

#         if ( ! s.factordone)
#             warning(fname, "matrix must be factored before", " calling this routine.")
#             warning("one - norm of a inverse not computed.")
#             return
#         end

#         if (n == 0)
#             warning(fname, "matrix is null.")
#             warning("one - norm of a inverse not computed.")
#             return
#         end

#         x(1:n) = one / n

#         triangularsolve(s, x)

#         if (n == 1)
#             est = abs(x(1));  return
#         end

#         est = sasum(n, x, 1)

#         for i = 1: n
#             x(i) = sign(one, x(i));  isgn(i) = nint(x(i))
#         end

#         transposetriangularsolve(s, x)

#         j = isamax(n, x, 1);   iter = 1

# #       Main loop - iterations 2, 3, ..., itmax.
#         do
#             iter = iter + 1;  x = zero;   x(j) = one

#             triangularsolve(s, x)

#             estold = est;  est = sasum(n, x, 1)

#             for i = 1: n
#                 if (nint(sign(one, x(i))) / = isgn(i)) FIXME goto 320
#             end

#             exit    #  Repeated sign vector, algorithm has converged.

#   320       continue
# #           Test for cycling.
#             if (est < = estold) exit

#             for i = 1: n
#                 x(i) = sign(one, x(i));  isgn(i) = nint(x(i))
#             end

#             transposetriangularsolve(s, x)

#             jlast = j;   j = isamax(n, x, 1)

#             if ((x(jlast) == abs(x(j))) || (iter > = itmax)) exit

#         end

# #       Iteration complete. Final stage.
#   410   continue
#         altsgn = one
#         for i = 1: n
#             x(i) = altsgn * (one + (i - one) / (n - one));  altsgn = - altsgn
#         end

#         triangularsolve(s, x)

#         temp = two * sasum(n, x, 1) / (3.0d0 * n)
#         if (temp > est) est = temp

#     end

# function sparsespdcondest(s, estcondnbr, p, mtype)
#         type (SparseSpdSolver) :: s
#         type(problem) :: p
#         real(double) :: estcondnbr
#         character (len = *), optional :: mtype
#         character (len = *), parameter :: fname = "sparsespdcondest:"
#         real(double) :: onenorma, onenormainv

#         trace (fname, "problem object name is", p.objectname)

#         if ( ! s.factordone)
#             warning(fname, "matrix must be factored before", " calling this routine.")
#             warning("condition number not computed.")
#             return
#         end

#         gettime(s.condesttime)

#         onenorma = onenorm(p, mtype)

#         sparsespdainverseonenorm(s, onenormainv)
#         estcondnbr = onenorma * onenormainv

#         gettime(s.condesttime, s.condesttime)

#         s.condestdone = true

#     end

end # module spksparsespdsolver
