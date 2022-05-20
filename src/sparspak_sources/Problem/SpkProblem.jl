"""
This is the "Problem" class -  - a major module of Sparspak90.

The variable "nRows"
- number of rows in the matrix
- number of elements in a row permutation
The variable "nCols"
- number of columns in the matrix
- number of elements in a column permutation
Other variables:

NOTE: a user can input (i, j, v), indicating that position
(i, j) in the matrix contains the value v.
Or, a user can input (i, j), indicating that position
(i, j) is nonzero, but not specifying a value there.
Hence the distinction between "nonzeros" and
"nonzero values" below.

nNZ - number of nonzero * values * in the matrix.
dNZ - number of nonzero * values * on the diagonal of the matrix.
nEdges - number of nonzeros in the matrix.
dEdges - number of nonzeros on the diagonal of the matrix.

The elements of the matrix are stored by columns using three ``parallel""
arrays:(link, rowSubs, values). The first element of column i stored is
found in values(head(i)). The row subscript of the element is
rowSubs(head(i)). The next element in the column is values(link(head(i)))
and so on. A zero value for link marks the end of the column.

NOTE: the elements in each column are stored in increasing
order of row subscript. Some algorithms used in the
package depend on this fact.

The right hand side of the matrix equation is stored in rhs,
and the solution (when provided or computed) is stored in the array x.
The size of the arrays is extended as required, and their lengths
for not generally correspond to the number of nonzeros in the matrix:
or the number of columns in the matrix.
    
    lenHead is the current length of the arrays head and x.
    lenRhs  is the current length of the array rhs.
    lenLink is the current length of the arrays link, rowSubs, values.
    
    lastUsed is the last position in values and link that is occupied.
    
    The user can improve efficiency by providing an estimate of the
    number of nonzeros in the matrix -  - this is done via the optional
    keyword parameter "NNZ" in the subroutine Construct.
    
    Similarly, the user can improve efficiency by providing
    estimates for the number of rows and columns in the matrix via the
        optional keyword parameters "nRows" and "nCols".
"""

module SpkProblem

mutable struct Problem{IT, FT}
    objectname::String
    info::String
    lenhead::IT
    lenlink::IT
    lenrhs::IT
    lastused::IT
    head::Vector{IT}
    link::Vector{IT}
    rowsubs::Vector{IT}
    nrows::IT
    ncols::IT
    nnz::IT
    dnz::IT
    nedges::IT
    dedges::IT
    values::Vector{FT}
    rscales::Vector{FT}
    cscales::Vector{FT}
    x::Vector{FT}
    rhs::Vector{FT}
end



"""
"""
function Problem(nrows::IT, ncols::IT, nnz::IT=2500, z::T=0.0, objectname="problem") where {IT, T}
    lenlink = nnz
    lenhead = ncols
    lenrhs = nrows

    nrows = zero(IT)
    ncols = zero(IT)
    lastused = zero(IT)
    nedges = zero(IT)
    dnz = zero(IT)
    dedges = zero(IT)
    nnz = zero(IT)

    info = ""
    head = fill(zero(IT), lenhead)
    rhs = fill(zero(T), lenrhs)
    x = fill(zero(T), lenhead)
    link = fill(zero(IT), lenlink)
    rowsubs = fill(zero(IT), lenlink)
    values = fill(_BIGGY, lenlink)

    return new(objectname, info, lenhead, lenlink, lenrhs, lastused, head,
        link, rowsubs, nrows, ncols, nnz, dnz, nedges, dedges, values,
        rscales, cscales, x, rhs)
end

end # module 




