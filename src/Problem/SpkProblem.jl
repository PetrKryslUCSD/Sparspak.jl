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

using SparseArrays
using ..SpkUtilities: _BIGGY

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
function Problem(nrows::IT, ncols::IT, nnz::IT=2500, z::FT=0.0, objectname="problem") where {IT, FT}
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
    rhs = fill(zero(FT), lenrhs)
    x = fill(zero(FT), lenhead)
    link = fill(zero(IT), lenlink)
    rowsubs = fill(zero(IT), lenlink)
    values = fill(_BIGGY(), lenlink)
    rscales = FT[]
    cscales = FT[]

    return Problem(objectname, info, lenhead, lenlink, lenrhs, lastused, head,
        link, rowsubs, nrows, ncols, nnz, dnz, nedges, dedges, values,
        rscales, cscales, x, rhs)
end

function inaij(p::Problem{IT,FT}, rnum, cnum, aij=zero(FT)) where {IT,FT}
    if (rnum < 1 || cnum < 1)
        @warn "$(@__FILE__): invalid matrix subscripts $(rnum), $(cnum): input ignored"
        return false
    end

    if (p.lastused > p.lenlink - 1)
        p.lenlink = max(2 * p.lenlink, 3 * p.ncols)
        p.link = extend(p.link, p.lenlink)
        p.rowsubs = extend(p.rowsubs, p.lenlink)
        p.values = extend(p.values, p.lenlink, _BIGGY)
    end

    p.nrows = max(rnum, p.nrows)
    p.ncols = max(p.ncols, cnum)

    if (p.ncols > p.lenhead)
        p.lenhead = 2 * p.ncols
        p.head = extend(p.head, p.lenhead)
        p.x = extend(p.x, p.lenhead)
    end

    if (p.nrows > p.lenrhs)
        p.lenrhs = 2 * p.nrows
        p.rhs = extend(p.rhs, p.lenrhs)
    end

    ptr = p.head[cnum]    # ^ * check if rnum is in the row list.
    lastptr = 0
    while (ptr > 0)
        if (p.rowsubs[ptr] > rnum)
            break
        end
        if (p.rowsubs[ptr] == rnum)
            if (p.values[ptr] == biggy)
                p.nnz = p.nnz + 1
                p.values[ptr] = aij
                if (rnum == cnum)
                    p.dnz = p.dnz + 1
                end
            else
                p.values[ptr] = p.values[ptr] + aij
            end
            return
        end
        lastptr = ptr
        ptr = p.link[ptr]
    end                 # ^ * not there; add to column

    p.lastused = p.lastused + 1
    p.nedges = p.nedges + 1
    p.rowsubs[p.lastused] = rnum

    if (rnum == cnum)
        p.dedges = p.dedges + 1
    end

    if (lastptr == 0)         # first element in the list end
        p.link[p.lastused] = p.head[cnum]
        p.head[cnum] = p.lastused
    elseif (ptr == 0)          # add to the end of the list end
        p.link[lastptr] = p.lastused
        p.link[p.lastused] = 0
    else                       # insert in the middle of the list.
        p.link[lastptr] = p.lastused
        p.link[p.lastused] = ptr
    end

    p.values[p.lastused] = aij
    p.nnz = p.nnz + 1
    if (rnum == cnum)
        p.dnz = p.dnz + 1
    end
    return true
end 

function inbi(p::Problem{IT, FT}, rnum::IT, bi::FT) where {IT, FT}
    if (rnum < 1)
        @error "$(@__FILE__): Invalid rhs subscript $(rnum)."
        return false
    end

    p.nrows = max(rnum, p.nrows)
    if (p.nrows > p.lenrhs)
        p.lenrhs = 2 * p.nrows; p.rhs = extend(p.rhs, p.lenrhs) 
    end

    p.rhs[rnum] = p.rhs[rnum] + bi 
    return true
end

function insparse(p::Problem{IT,FT}, spm) where {IT,FT}
    I, J, V = findnz(spm)
    for i in eachindex(I)
        if !inaij(p, I[i], J[i], V[i])
            return false 
        end
    end
    return true
end

function outsparse(p::Problem{IT,FT})  where {IT,FT}
    if (p.nrows == 0 && p.ncols == 0) 
        return spzeros(p.nrows, p.ncols)
    end
    nr = p.nrows;  nc = p.ncols
    I = IT[]
    J = IT[]
    V = FT[]
    for i in 1:nc
        ptr = p.head[i]
        while (ptr > 0)
            r = p.rowsubs[ptr]
            if (r <= nr)
                push!(J, i)
                push!(I, r)
                push!(V, p.values[ptr])
            end
            ptr = p.link[ptr]
        end
    end
    return sparse(I, J, V, nr, nc)
end

end # module 




