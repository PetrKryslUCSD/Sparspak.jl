""" 
Ordering class:

nRows is the number of rows in the matrix
nCols is the number of columns in the matrix

Ordering objects contain two permutations and their inverses:

rPerm is a row permutation: rPerm(i) = k means that the new position of
    row k is in position i in the new permutation.
rInvp is a row permutation satisfying rInvp(rPerm(i)) = i. Thus, rInvp(k)
    provides the position in the new ordering of the original row k.

cPerm and cInvp are analogous to rPerm and rInvp, except they apply
    to column permutations of the matrix.

When the matrix is symmetrically permuted, rPerm = cPerm and rInvp = cInvp.

xRowBlk  is an array that is sometimes used to contain a partitioning
    of the rows of the matrix, or both the rows and columns when the
    matrix is symmetric.

nRowBlks is the number of blocks in the partitioning; the rows of the
    i - th partition are xRowBlk(i), xRowBlk(i) + 1 ... xRowBlk(i + 1) - 1.0
    For convenience, xRowBlk has nRowBlks + 1 elements, with
    xRowBlk(nRowBlks + 1) = nRows + 1.0

    When the matrix is not symmetric, and a partitioning of the columns
     is required as well, the pair (nColBlks, xColBlk) is used.
"""
module SpkOrdering

"""
    Ordering{IT}

Type of ordering of the rows and columns.
"""
mutable struct Ordering{IT}
    nrows::IT
    ncols::IT
    nrowblks::IT
    ncolblks::IT
    rperm::Vector{IT}
    rinvp::Vector{IT}
    cperm::Vector{IT}
    cinvp::Vector{IT}
    xrowblk::Vector{IT}
    xcolblk::Vector{IT}
end

"""
    Ordering(nrows::IT) where {IT}

Construct an ordering object. 

Since only one parameter(`nrows`) is supplied, it is assumed that the size of the
row ordering and column ordering are the same. That is, that the matrix is
square.

Input Parameters:
  order - the ordering (declared in the calling program)
  nRows - the number of rows (and columns) in the matrix
  objectName - the name of the ordering object (optional)
"""
function Ordering(nrows::IT) where {IT}
    return Ordering(nrows, nrows)
end

"""
    Ordering(nrows::IT, ncols::IT) where {IT}

Construct an ordering object. 

The arrays `rperm`, `cperm`, `rinvp`, `cinvp` are allocated and initialized to the
identity permutation.

Input Parameter:
  order - the ordering (declared in the calling program)
  nRows, nCols - the number of rows and columns in the matrix
  objectName - the name of the ordering object (optional)
"""
function Ordering(nrows::IT, ncols::IT) where {IT}
    #       The default is to set nRowBlks and nColBlks to 0.
    nrowblks = 0
    ncolblks = 0

    rperm = fill(zero(IT), nrows)
    rinvp = fill(zero(IT), nrows)
    cperm = fill(zero(IT), ncols)
    cinvp = fill(zero(IT), ncols)

    rperm .= 1:nrows
    rinvp .= rperm

    cperm .= 1:ncols
    cinvp .= cperm
    xrowblk = IT[]
    xcolblk = IT[]

    return Ordering(nrows, ncols, nrowblks, ncolblks, rperm, rinvp, cperm, cinvp, xrowblk, xcolblk)
end

end