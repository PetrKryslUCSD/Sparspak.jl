# _SparseBase class:
# This class contains the variables for the data structure for the
# SparseSolver.
# #
# The Cholesky factor of the matrix is stored column by column.
# Only the nonzeros of the matrix are stored, in an efficient way by
# exploiting supernode structure. A supernode is simply a set of
# consecutive columns, say columns i, i + 1, i + 2, ... i + s - 1, all of which
# have essentially identical structure. Column i and columns i + 1
# have nonzeros in identical rows, except that element (i, i + 1) is
# zero (of course). If one stores the subscripts of the nonzero
# elements for column i,  one can easily determine the subscripts of
# the nonzeros in the other columns of the supernode.
# #
# n - the number of rows and columns in the matrix.
# #
# nnzl - the number of nonzeros in the Cholesky factor L.
# #
# nsub - the number of row subscripts stored. This is also the
#               length of the array lindx, and equal to xlindx(n + 1) - 1.0
# #
# nsuper - the number of supernodes in the Cholesky factor
# #
# xsuper - the supernode structure. The i - th supernode consists
#               of columns xsuper(i), xsuper(i) + 1, ... xsuper(i + 1) - 1.0
#               ie.  maps a supernode to it"s first column
# #
# errflag - an error flag. If a zero diagonal element is detected
#               during the factorization, errflag is set to - 1.0
#               If no errors are detected, errflag will be 0.0
# #
# (xlindx, lindx) - a pair of arrays containing the row subscripts of
#               the first columns of the supernodes. The row subscripts
#               of the nonzeros in the i - th supernode (including the
#               diagonal element) are given by lindx(xlindx(i)),
#               lindx(xlindx(i) + 1), ..., lindx(xlindx(i + 1) - 1)
#               Note that lindx is of length nsub. The array
#               xlindx is of length nsuper + 1, with xlindx(nsuper + 1)
#               equal to nsub + 1.0
#               This data structure also contains the column subscripts
#               of U; however, they begin at lindx(xlindx(jsup) + width(jsup))
#               where width(jsup) = xsuper(jsup + 1) - xsuper(jsup)
# #
# (xlnz, lnz) - the nonzero elements of the Cholesky factor. The nonzero
#               elements in the k - th column of L are given by
#               lnz(xlnz(k)), lnz(xlnz(k) + 1), ...,
#               lnz(xlnz(xsuper(snode(k + 1)) - 1).
# (xunz, unz) - the nonzero elements of the Cholesky factor. The nonzero
#               elements in the k - th row of U are given by
#               lnz(xlnz(k)), lnz(xlnz(k + 1), ...,
#               lnz(xlnz(xsuper(snode(k) + 1)) - 1)
# #
# colcnt - array of length neqns, containing the number
#               of nonzeros in each column of the factor,
# #
# The algorithm employed is a so - called "left - looking" scheme; each
# supernode is computed by applying modifications from other supernodes
# to its left.
# #
# snode - This is an array which makes it convenient to quickly
#               determine in which supernode a column resides.
#               snode(i) = k means column i is in supernode k.
# #
# maxBlockSize - in order to improve multiprocessor utilization, it is
#                helpful to split the larger supernodes into smaller ones.
#                This is done by introducing artificial supernodes.
#                The variable maxBlockSize governs this process; no
#                supernode will have more than maxBlockSize columns.
#                At present, this is fixed at 30 columns.
# #
# ipiv - This array is used to hold the piviting information
#                about the rows of U and the diagonal blocks.
#                This information will be needed later to solve the system.
# #
# tempSizeNeed - To use the BLAS routines, a temporary array will be
#               need to put calculations into.  The maximum size needed
#               is the max number of non zero"s in any supernode.
#               This calculation is done inside the factorization routine
#               and can pretty much be removed from here...
#               For whatever reason, the calculations here and in the
#               factorization sseem to differ... bummer#
# #
# For more details on the algorithm and storage scheme used in this solver
# class, please refer to Chapter 5 of the book "Computer Solution of Large
# Sparse Positive Definite Systems" by Alan George and Joseph Liu published
# by Prentice - Hall, 1981.0

module SpkSparseBase

using ..SpkOrdering: Ordering
using ..SpkGraph: Graph, makestructuresymmetric
using ..SpkETree: ETree, _getetree!, _getpostorder!
using ..SpkSymfct: _findcolumncounts!, _symbolicfact!, _findsupernodes!
using ..SpkLUFactor: _lufactor!, _lulsolve!, _luusolve!
using ..SpkProblem: Problem
using ..SpkUtilities: __extend
using ..SpkMmd: mmd!

mutable struct _SparseBase{IT, FT}
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
    tempsizeneed::IT
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
    xunz::Vector{IT}
    ipiv::Vector{IT}
    lnz::Vector{FT}
    unz::Vector{FT}
end


# This routine initializes the solver object s by retrieving some
# information from the problem object p. The solver object contains
# an elimination tree and an ordering object; these are also
#  initialized in this routine.
function _SparseBase(p::Problem{IT,FT}) where {IT,FT}
    maxblocksize = 30   # This can be set by the user

    tempsizeneed = zero(IT)
    n = p.nrows
    nnz = p.nnz
    nnzl = zero(IT)
    nsub = zero(IT)

    nsuper = zero(IT)
    if (n > zero(IT))
        nsuper = 1
    end

    factorops = zero(FT)
    solveops = zero(FT)
    realstore = zero(FT)
    integerstore = zero(FT)
    errflag = 0

    order = Ordering(n)   # ordering object for the solver
    g = Graph(p)
    t = ETree(n)

    colcnt = IT[]
    snode = IT[]
    xsuper = IT[]
    xlindx = IT[]
    lindx = IT[]
    xlnz = IT[]
    xunz = IT[]
    ipiv = IT[]
    lnz = FT[]
    unz = FT[]

    return _SparseBase(order, t, g, errflag, n, nnz, nnzl, nsub, nsuper, maxblocksize,
        tempsizeneed, factorops, solveops, realstore, integerstore,
        colcnt, snode, xsuper, xlindx, lindx, xlnz, xunz, ipiv,
        lnz, unz)
end

function _findorder!(s::_SparseBase{IT}, orderfunction::F) where {IT, F}
    if (s.n == 0)
        error("An empty problem, no ordering found.")
        return false
    end
    makestructuresymmetric(s.g)     # Make it symmetric
    orderfunction(s.g, s.order)   # Default ordering function
    return true
end

function _findorder!(s::_SparseBase{IT}) where {IT}
    return _findorder!(s, mmd!)
end

function _symbolicfactor!(s::_SparseBase{IT, FT}) where {IT, FT}
# """
#     This subroutine computes the storage requirements and sets up
#     data structures for the symbolic and numerical factorization.
##
# Input parameters: solver object
##
# The "output" is the modified solver object.
#
#
# """
    if (s.n == 0)
        error("An empty problem. No symbolic factorization done.")
        return false
    end

    s.colcnt = fill(zero(IT), s.n)
    s.snode = fill(zero(IT), s.n)
    s.xsuper = fill(zero(IT), s.n + 1)

# -  -  -  -  -  -  -  -  -
#       Compute elimination tree
# -  -
    _getetree!(s.g, s.order, s.t)
    _getpostorder!(s.t, s.order)

# -
#       Compute row and column factor nonzero counts.
# -
    _findcolumncounts!(s.g.nv, s.g.xadj, s.g.adj, s.order.rperm, s.order.rinvp, s.t.parent, s.colcnt, s.nnzl)
    _getpostorder!(s.t, s.order, s.colcnt)

#-  -    -
#       Find supernodes. Split them so none are larger than maxBlockSize
# -  -  -
    s.nsub, s.nsuper = _findsupernodes!(s.g.nv, s.t.parent, s.colcnt, s.nsub, s.nsuper, s.xsuper, s.snode, s.maxblocksize)
    s.xsuper = __extend(s.xsuper, s.nsuper + 1)

    s.lindx = fill(zero(IT), s.nsub)
    s.xlindx = fill(zero(IT), s.nsuper + 1)

# -
#       setup for symbolic factorization.
# -
    s.xlnz = fill(zero(IT), s.n + 1)
    s.xunz = fill(zero(IT), s.n + 1)
    s.ipiv = fill(zero(IT), s.n)

# -
#       Set up the data structure for the Cholesky factor.
# -
    findnonzeroindexs(s.n, s.colcnt, s.nsuper, s.xsuper, s.xlnz, s.xunz, s.tempsizeneed)
    
    _symbolicfact!(s.g.nv, s.g.xadj, s.g.adj, s.order.rperm, s.order.rinvp, s.colcnt, s.nsuper, s.xsuper, s.snode, s.nsub, s.xlindx, s.lindx)

# -
#       We now know how many elements we need, so allocate for it.
# -
    s.lnz = fill(zero(FT), s.xlnz[s.n + 1] - 1)
    s.unz = fill(zero(FT), s.xunz[s.n + 1] - 1)


    s.lnz[1:s.xlnz[s.n + 1] - 1] .= zero(FT)
    s.unz[1:s.xunz[s.n + 1] - 1] .= zero(FT)
    s.ipiv[1:s.n] .= zero(IT)

    return true
end

function findnonzeroindexs(n, colcnt, nsuper, xsuper, xlnz, xunz, maxtemp)
# """
#    This routine inserts the index offsets for the arrays xlnz and xunz.
# """
#
        # integer :: nsuper, maxtemp, n
        # integer :: xsuper(*)
        # integer :: colcnt(*), xlnz(*), xunz(*)
        # integer :: fstcol, ksup, lstcol, point, width, nnzinsuper
        # integer :: upoint, jcol

    upoint = 1; point = 1; nnzinsuper = 0

    for ksup in 1:nsuper
        fstcol = xsuper[ksup]; lstcol = xsuper[ksup + 1] - 1
        maxtemp = max(maxtemp, nnzinsuper)
        nnzinsuper = 0
        width = lstcol - fstcol + 1
        for jcol in fstcol:lstcol
            xlnz[jcol] = point
            xunz[jcol] = upoint
            nnzinsuper = nnzinsuper + colcnt[fstcol]
            point = point + colcnt[fstcol]
            upoint = upoint + colcnt[fstcol] - width
        end
        nnzinsuper = nnzinsuper + colcnt[fstcol]
    end

    xlnz[n + 1] = point; point = 1
    xunz[n + 1] = upoint

end


#     This subroutine retrieves a matrix from a problem object p and
#     inserts its elements into the data structure for the Cholesky
#     factor L. The data structure involves the arrays xlindx, lindx,
#     xlnz, lnz, xunz and unz as described in the comments at the
#     beginning of this module.
#     The components of the problem object are decribed in the
#     comments at the beginning of the SpkProblem module.
#     Note that the Cholesky factor L corresponds to the matrix
#     problem after it has had its rows and columns permuted.
#     Thus, a reordering is applied to its elements before they are
#     inserted into the data structure.
# 
# Input parameters: problem and solver objects
# The "output" is the modified solver object.

function _inmatrix!(s::_SparseBase{IT, FT}, p::Problem{IT, FT}) where {IT, FT}
    if (s.n == 0)
        error("An empty problem. No matrix.")
        return false
    end

    s.lnz .= zero(FT)
    s.unz .= zero(FT)
    s.ipiv .= zero(IT)

    function doit(ncols, link, head, rinvp, cinvp, rowsubs, snode, xsuper, xlindx, lindx, values, xlnz, lnz, xunz, unz)
        for i in 1:ncols
            ptr = head[i]
            while (ptr > 0)  # scan column i ....
                inew = rinvp[rowsubs[ptr]];
                jnew = cinvp[i]
                value = values[ptr];

                if (inew >= xsuper[snode[jnew]])
#               Lies in L.  get pointers and lengths needed to search
#               column jnew of L for location l(inew, jnew).
                    jsup = snode[jnew];
                    fstcol = xsuper[jsup]
                    fstsub = xlindx[jsup]
                    lstsub = xlindx[jsup + 1] - 1
                    nnzloc = 0;
                    for nxtsub in fstsub:lstsub
                        irow = lindx[nxtsub]
                        if  (irow > inew)
                            error("No space for matrix element ($(inew), $(jnew)).")
                            return false
                        end
                        if  (irow == inew)
#                       find a proper offset into lnz and increment by value
                            _p = xlnz[jnew] + nnzloc
                            lnz[_p] += value
                            break
                        end
                        nnzloc = nnzloc + 1
                    end
                else
#               Lies in U
                    jsup = snode[inew]
                    fstcol = xsuper[jsup]
                    lstcol = xsuper[jsup + 1] - 1
                    width = lstcol - fstcol + 1
                    lstsub = xlindx[jsup + 1] - 1
                    fstsub = xlindx[jsup] + width
                    nnzloc = 0;
                    for nxtsub in fstsub:lstsub
                        irow = lindx[nxtsub]
                        if  (irow > jnew)
                            error("No space for matrix element ($(inew), $(jnew)).")
                            return false
                        end
                        if  (irow == jnew)
#                       find a proper offset into unz and increment by value
                            _p = xunz[inew] + nnzloc
                            unz[_p] += value
                            break
                        end
                        nnzloc = nnzloc + 1
                    end
                end
                ptr = link[ptr]
            end
        end
        return true
    end
    return doit(p.ncols, p.link, p.head, s.order.rinvp, s.order.cinvp, p.rowsubs, s.snode, s.xsuper, s.xlindx, s.lindx, p.values, s.xlnz, s.lnz, s.xunz, s.unz)
end

# This routine calls the lower - level routine LUFactor which
# computes and L U factorization of the matrix stored in the
# solver object s. The components of the solver object are described
#  in the comments at the beginning of this module.
function _factor!(s::_SparseBase{IT, FT}) where {IT, FT}
    if (s.n == 0)
        error("An empty problem. No matrix.")
        return false
    end

    s.errflag = _lufactor!(s.n, s.nsuper, s.xsuper, s.snode, s.xlindx, s.lindx, s.xlnz, s.lnz, s.xunz, s.unz, s.ipiv)

    if (s.errflag != 0)
        error("An empty problem. No matrix.")
        return false
    end
    return true
end


# This routine calls the lower - level routine LUSolve which
# solves L U x = rhs, given L, U and rhs. The solution is placed in
# the array rhs. Note that the factorization is of a permuted form of
# the problem, and it is assumed that the user provides the rhs in
# the original order. Thus, the permutation "hidden" in the solver
#  object is applied as appropriate.
function _triangularsolve!(s::_SparseBase{IT, FT}, solution::AbstractVector{FT}) where {IT, FT}
    if (s.n == 0)
        error("An empty problem. No solution.")
        return false
    end

    rhs = fill(zero(FT), s.n)
    rhs .= solution[s.order.rperm]

    _lulsolve!(s.nsuper, s.xsuper, s.xlindx, s.lindx, s.xlnz, s.lnz, s.ipiv, rhs)

    _luusolve!(s.n, s.nsuper, s.xsuper, s.xlindx, s.lindx, s.xlnz, s.lnz, s.xunz, s.unz, rhs)

    solution .= rhs[s.order.rinvp]

    return true
end

end
