""" SparseBase class:
This class contains the variables for the data structure for the
SparseSolver.
#
The Cholesky factor of the matrix is stored column by column.
Only the nonzeros of the matrix are stored, in an efficient way by
exploiting supernode structure. A supernode is simply a set of
consecutive columns, say columns i, i + 1, i + 2, ... i + s - 1, all of which
have essentially identical structure. Column i and columns i + 1
have nonzeros in identical rows, except that element (i, i + 1) is
zero (of course). If one stores the subscripts of the nonzero
elements for column i,  one can easily determine the subscripts of
the nonzeros in the other columns of the supernode.
#
n - the number of rows and columns in the matrix.
#
nnzl - the number of nonzeros in the Cholesky factor L.
#
nsub - the number of row subscripts stored. This is also the
              length of the array lindx, and equal to xlindx(n + 1) - 1.0
#
nsuper - the number of supernodes in the Cholesky factor
#
xsuper - the supernode structure. The i - th supernode consists
              of columns xsuper(i), xsuper(i) + 1, ... xsuper(i + 1) - 1.0
              ie.  maps a supernode to it"s first column
#
errflag - an error flag. If a zero diagonal element is detected
              during the factorization, errflag is set to - 1.0
              If no errors are detected, errflag will be 0.0
#
(xlindx, lindx) - a pair of arrays containing the row subscripts of
              the first columns of the supernodes. The row subscripts
              of the nonzeros in the i - th supernode (including the
              diagonal element) are given by lindx(xlindx(i)),
              lindx(xlindx(i) + 1), ..., lindx(xlindx(i + 1) - 1)
              Note that lindx is of length nsub. The array
              xlindx is of length nsuper + 1, with xlindx(nsuper + 1)
              equal to nsub + 1.0
              This data structure also contains the column subscripts
              of U; however, they begin at lindx(xlindx(jsup) + width(jsup))
              where width(jsup) = xsuper(jsup + 1) - xsuper(jsup)
#
(xlnz, lnz) - the nonzero elements of the Cholesky factor. The nonzero
              elements in the k - th column of L are given by
              lnz(xlnz(k)), lnz(xlnz(k) + 1), ...,
              lnz(xlnz(xsuper(snode(k + 1)) - 1).
(xunz, unz) - the nonzero elements of the Cholesky factor. The nonzero
              elements in the k - th row of U are given by
              lnz(xlnz(k)), lnz(xlnz(k + 1), ...,
              lnz(xlnz(xsuper(snode(k) + 1)) - 1)
#
colcnt - array of length neqns, containing the number
              of nonzeros in each column of the factor,
#
The algorithm employed is a so - called "left - looking" scheme; each
supernode is computed by applying modifications from other supernodes
to its left.
#
snode - This is an array which makes it convenient to quickly
              determine in which supernode a column resides.
              snode(i) = k means column i is in supernode k.
#
maxBlockSize - in order to improve multiprocessor utilization, it is
               helpful to split the larger supernodes into smaller ones.
               This is done by introducing artificial supernodes.
               The variable maxBlockSize governs this process; no
               supernode will have more than maxBlockSize columns.
               At present, this is fixed at 30 columns.
#
ipiv - This array is used to hold the piviting information
               about the rows of U and the diagonal blocks.
               This information will be needed later to solve the system.
#
tempSizeNeed - To use the BLAS routines, a temporary array will be
              need to put calculations into.  The maximum size needed
              is the max number of non zero"s in any supernode.
              This calculation is done inside the factorization routine
              and can pretty much be removed from here...
              For whatever reason, the calculations here and in the
              factorization sseem to differ... bummer#
#
For more details on the algorithm and storage scheme used in this solver
class, please refer to Chapter 5 of the book "Computer Solution of Large
Sparse Positive Definite Systems" by Alan George and Joseph Liu published
by Prentice - Hall, 1981.0


"""
module SpkSparseBase

using ..SpkOrdering: Ordering
using ..SpkGraph: Graph
using ..SpkETree: ETree
using ..SpkProblem: Problem

mutable struct SparseBase{IT, FT}
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

"""
This routine initializes the solver object s by retrieving some
information from the problem object p. The solver object contains
an elimination tree and an ordering object; these are also
 initialized in this routine.
"""
function SparseBase(p::Problem{IT,FT}) where {IT,FT}
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

    return SparseBase(order, t, g, errflag, n, nnz, nnzl, nsub, nsuper, maxblocksize,
        tempsizeneed, factorops, solveops, realstore, integerstore,
        colcnt, snode, xsuper, xlindx, lindx, xlnz, xunz, ipiv,
        lnz, unz)
end

end