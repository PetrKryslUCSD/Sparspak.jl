
""" SparseSpdBase class:
This class contains the variables for the data structure for the
SparseSpdSolver.

The Cholesky factor of the matrix is stored column by column.
Only the nonzeros of the matrix are stored, in an efficient way by
exploiting supernode structure. A supernode is simply a set of
consecutive columns, say columns i, i + 1, i + 2, ... i + s - 1, all of which
have essentially identical structure. Column i and columns i + 1
have nonzeros in identical rows, except that element (i, i + 1) is
zero (of course). If one stores the subscripts of the nonzero
elements for column i,  one can easily determine the subscripts of
the nonzeros in the other columns of the supernode.

n - the number of rows and columns in the matrix.

nnzl - the number of nonzeros in the Cholesky factor L.

nsub - the number of row subscripts stored. This is also the
              length of the array lindx, and equal to xlindx(n + 1) - 1.0

nsuper - the number of supernodes in the Cholesky factor

xsuper - the supernode structure. The i - th supernode consists
              of columns xsuper(i), xsuper(i) + 1, ... xsuper(i + 1) - 1.0
              ie.  maps a supernode to it"s first column

errflag - an error flag. If a zero diagonal element is detected
              during the factorization, errflag is set to - 1.0
              If no errors are detected, errflag will be 0.0

(xlindx, lindx) - a pair of arrays containing the row subscripts of
              the first columns of the supernodes. The row subscripts
              of the nonzeros in the i - th supernode (including the
              diagonal element) are given by lindx(xlindx(i)),
              lindx(xlindx(i) + 1), ..., lindx(xlindx(i + 1) - 1)
              Note that lindx is of length nsub. The array
              xlindx is of length nsuper + 1, with xlindx(nsuper + 1)
              equal to nsub + 1.0

(xlnz, lnz) - the nonzero elements of the Cholesky factor. The nonzero
              elements in the k - th column of L are given by
              lnz(xlnz(k)), lnz(xlnz(k) + 1), ..., lnz(xlnz(k + 1) - 1).

colcnt - array of length neqns, containing the number
              of nonzeros in each column of the factor,

The algorithm employed is a so - called "left - looking" scheme; each
supernode is computed by applying modifications from other supernodes
to its left.

snode - This is an array which makes it convenient to quickly
              determine in which supernode a column resides.
              snode(i) = k means column i is in supernode k.


maxBlockSize - in order to improve multiprocessor utilization, it is
              helpful to split the larger supernodes into smaller ones.
              This is done by introducing artificial supernodes.
              The variable maxBlockSize governs this process; no
              supernode will have more than maxBlockSize columns.
              At present, this is fixed at 60 columns, as it seems
              to be optimal for problems upto 700^2

For more details on the algorithm and storage scheme used in this solver
class, please refer to Chapter 5 of the book "Computer Solution of Large
Sparse Positive Definite Systems" by Alan George and Joseph Liu published
by Prentice - Hall, 1981.0


"""
module SpkSparseSpdBase

using ..SpkOrdering: Ordering
using ..SpkGraph: Graph, makestructuresymmetric
using ..SpkETree: ETree, _getetree!, _getpostorder!
using ..SpkMmd: mmd!
using ..SpkProblem: Problem
using ..SpkSymfct: _findcolumncounts!, _symbolicfact!, _findsupernodes!
using ..SpkUtilities: __extend

mutable struct _SparseSpdBase{IT, FT}
    order::Ordering
    t::ETree
    g::Graph
    errflag::IT
    n::IT
    nnz::IT
    nnzl::IT
    nsub::IT
    nsuper::IT
    maxblocksize::IT
    factorops::FT
    solveops::FT
    realstore::FT
    integerstore::FT
    colcnt::Vector{IT}
    snode::Vector{IT}
    xsuper::Vector{IT}
    xlindx::Vector{IT}
    lindx::Vector{IT}
    xlnz::Vector{IT}
    lnz::Vector{FT}
end

"""
"""
function _SparseSpdBase(p::Problem{IT,FT}) where {IT,FT}
    maxblocksize = 60;   # This can be set by the user

    n = p.nrows; nnz = p.nnz; nnzl = zero(IT)
    nsub = zero(IT);

    if (n > 0)
        nsuper = zero(IT) + 1
    else
        nsuper = zero(IT)
    end

    factorops = zero(FT); solveops = zero(FT); realstore = zero(FT); integerstore = zero(FT)
    errflag = zero(IT)

    order = Ordering(n)   # ordering object for the solver
    g = Graph(p)
    t = ETree(n)

    colcnt = IT[]
    snode = IT[]
    xsuper = IT[]
    xlindx = IT[]
    lindx = IT[]
    xlnz = IT[]
    ipiv = IT[]
    lnz = FT[]

    return _SparseSpdBase(order, t, g, errflag, n, nnz, nnzl, nsub, nsuper, maxblocksize, factorops, solveops, realstore, integerstore, colcnt, snode, xsuper, xlindx, lindx, xlnz, lnz)
end

function _findorder!(s::_SparseSpdBase{IT}, orderfunction::F) where {IT, F}
    if (s.n == 0)
        error("An empty problem, no ordering found.")
        return false
    end
    makestructuresymmetric(s.g)     # Make it symmetric
    orderfunction(s.g, s.order)   # Default ordering function
    return true
end

function _findorder!(s::_SparseSpdBase{IT}) where {IT}
    return _findorder!(s, mmd!)
end

# function SparseSpdBasefindorderperm(s, perm)
#         type (SparseSpdBase)  ::  s
#         integer :: perm(*)
#         character (len = *), parameter :: fname = "SparseSpdBasefindorderperm:"
#         integer :: i

#         trace (fname, "object name is", s.objectname)

#         if (s.n == 0)
#             warning(fname, "this solver object is derived from")
#             warning("an empty problem. no ordering to be found.")
#             return
#         end

#         makestructuresymmetric(s.g)     # Make it symmetric

#         s.order.rperm(1:s.n) = perm(1:s.n)
#         s.order.cperm(1:s.n) = perm(1:s.n)
#         s.order.rinvp(s.order.rperm(1:s.n)) = (/ (i, i = 1, s.n) /)
#         s.order.cinvp(s.order.cperm(1:s.n)) = (/ (i, i = 1, s.n) /)

#     end

function _symbolicfactor!(s::_SparseSpdBase{IT, FT}) where {IT, FT}

    if (s.n == 0)
        error("An empty problem. No symbolic factorization done.")
        return false
    end

    s.colcnt = zeros(IT, s.n)
    s.snode = zeros(IT, s.n)
    s.xsuper = zeros(IT, s.n + 1)

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#       Compute elimination tree
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    _getetree!(s.g, s.order, s.t)
    _getpostorder!(s.t, s.order)

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#       Compute row and column factor nonzero counts.
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    _findcolumncounts!(s.g.nv, s.g.xadj, s.g.adj, s.order.rperm, s.order.rinvp, s.t.parent, s.colcnt, s.nnzl)
    _getpostorder!(s.t, s.order, s.colcnt)

#-
#       Find supernodes. Split them so none are larger than maxBlockSize
#-
    s.nsub, s.nsuper = _findsupernodes!(s.g.nv, s.t.parent, s.colcnt, s.nsub, s.nsuper, s.xsuper, s.snode, s.maxblocksize)
    s.xsuper = __extend(s.xsuper, s.nsuper + 1)

    s.lindx = zeros(IT, s.nsub)
    s.xlindx = zeros(IT, s.nsuper + 1)

#-
#       NEXT LINE CHANGED SINCE LNZ NOW CONTAINS RECTANGULAR SUPERNODES,
#       NOT TRAPEZOIDAL, ie. some 0 elements are going to be stored
#-
    s.xlnz = zeros(IT, s.n + 1)

# -  -  -  -  -  -  -
#       Set up the data structure for the Cholesky factor.
# -
    _findnonzeroindexs!(s.n, s.colcnt, s.nsuper, s.xsuper, s.xlnz)

    _symbolicfact!(s.g.nv, s.g.xadj, s.g.adj, s.order.rperm, s.order.rinvp, s.colcnt, s.nsuper, s.xsuper, s.snode, s.nsub, s.xlindx, s.lindx)

# -  -  -  -  -  - -
#       We now know how many elements we need, so allocate for it.
#-
    s.lnz = zeros(FT, s.xlnz[s.n + 1])


    s.lnz[1:s.xlnz[s.n + 1]] .= zero(FT)

    return true
end

function _inmatrix!(s::_SparseSpdBase{IT, FT}, p::Problem{IT, FT}) where {IT, FT}
# """
#     This subroutine retrieves a matrix from a problem object p and
#     inserts its elements into the data structure for the Cholesky
#     factor L. The data structure involves the arrays xlindx, lindx,
#     xlnz, and lnz as described in the comments at the beginning of
#     this module. The components of the problem object are
#     decribed in the comments at the beginning of the SpkProblem
#     module. Note that the Cholesky factor L corresponds to the
#     matrix problem after it has had its rows and columns permuted.
#     Thus, a reordering is applied to its elements before they are
#     inserted into the data structure.
##
# Input parameters: problem and solver objects
#
# The "output" is the modified solver object.
##
# Working Parameters:
#     supcol - Will stores which column of its supernode a particular
#                  column will reside in.
#
#
# """
        # type (SparseSpdBase)  ::  s
        # type (problem)  ::  p
        # type (ordering) ::  order
        # character (len = *), parameter :: fname = "SparseSpdBaseinmatrix:"
        # integer :: rnum, cnum, i, k, ptr, nnzloc, nxtsub, irow
        # integer :: inew, jnew, itemp, jsup, fstcol, fstsub, lstsub
        # integer :: supcol, lnzoff
        # real (double) :: value

    if (s.n == 0)
        error("An empty problem. No matrix.")
        return false
    end

    s.lnz[1:s.xlnz(s.n + 1) - 1] .= zero(FT)


    for i  in 1: p.ncols           #   Only the lower triangle of p is used.
        ptr = p.head[i]
        while (ptr > 0)  # scan column i ....
            if (p.rowsubs[ptr] >= i)
                inew = s.order.rinvp[p.rowsubs[ptr]];
                value = p.values[ptr]; jnew = s.order.cinvp[i]
                if  (inew < jnew)
                    itemp = inew; inew = jnew; jnew = itemp
                end
#             get pointers and lengths needed to search column jnew
#                   of L for location l(inew, jnew).
                jsup = s.snode[jnew]; fstcol = s.xsuper[jsup]
                fstsub = s.xlindx[jsup] + jnew - fstcol
                lstsub = s.xlindx[jsup + 1] - 1
#           search for row subscript inew in jnew's subscript list.
                for nxtsub  in  fstsub: lstsub
                    irow = s.lindx[nxtsub]
                    if  (irow > inew)
                        error("No space for matrix element ($(inew), $(jnew)).")
                        return false
                    end

                    if  (irow == inew)
#                   find a proper offset into lnz.  lnz new stores
#                     rectangular columns, not trapezoidal columns
                        supcol = jnew - s.xsuper[s.snode[jnew]]
                        nnzloc = s.xlnz[jnew] + (nxtsub - fstsub) + supcol

                        s.lnz[nnzloc] = s.lnz[nnzloc] + value
                        break
                    end
                end
            end
            ptr = p.link[ptr]
        end
    end
    return true
end

function _factor!(s::_SparseSpdBase{IT, FT}) where {IT, FT}
    # """
    # This routine calls the lower - level routine LDLtFactor which
    # computes and L D L^T factorization of the matrix stored in the
    # solver object s. The components of the solver object are described
    #  in the comments at the beginning of this module.
    # """
    if (s.n == 0)
        error("An empty problem. No matrix.")
        return false
    end

    s.errflag = _ldltfactor!(s.n, s.nsuper, s.xsuper, s.snode, s.xlindx, s.lindx, s.xlnz, s.lnz)

    if (s.errflag != 0)
        error("An empty problem. No matrix.")
        return false
    end
    return true
end

function solve!(s::_SparseSpdBase{IT, FT}, solution::AbstractVector{FT}) where {IT, FT}

# This routine calls the lower - level routine LDLtSolve which
# solves L D L^T x = rhs, given L, D and rhs. The solution is placed in
# the array rhs. Note that the factorization is of a permuted form of
# the problem, and it is assumed that the user provides the rhs in
# the original order. Thus, the permutation "hidden" in the solver
#  object is applied as appropriate.

    if (s.n == 0)
        error("An empty problem. No solution.")
        return false
    end

    rhs .= solution[s.order.rperm]

    _ldltsolve!(s.nsuper, s.xsuper, s.xlindx, s.lindx, s.xlnz, s.lnz, rhs)

    solution .= rhs[s.order.rinvp]

    return true
end


    function _findnonzeroindexs!(n, colcnt, nsuper, xsuper, xlnz)
# """
#    This routine inserts the index offsets for the arrays xlnz and xunz.
# """

        point = 1
        for ksup  in  1: nsuper
            fstcol = xsuper[ksup];   lstcol = xsuper[ksup + 1] - 1
            for jcol  in  fstcol: lstcol
                xlnz[jcol] = point
                point = point + colcnt[fstcol]
            end
        end

        xlnz[n + 1] = point
        nothing
    end
#
end # module spkSparseSpdBase
