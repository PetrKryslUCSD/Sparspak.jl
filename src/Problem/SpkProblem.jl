# This is the `Problem` class -- a major module of Sparspak90.

# The variable `nrows`
# - number of rows in the matrix
# - number of elements in a row permutation
# The variable `ncols`
# - number of columns in the matrix
# - number of elements in a column permutation
# Other variables:

# NOTE: a user can input (i, j, v), indicating that position
# (i, j) in the matrix contains the value v.
# Or, a user can input (i, j), indicating that position
# (i, j) is nonzero, but not specifying a value there.
# Hence the distinction between "nonzeros" and
# "nonzero values" below.

# nnz - number of nonzero * values * in the matrix.
# dnz - number of nonzero * values * on the diagonal of the matrix.
# nedges - number of nonzeros in the matrix.
# dedges - number of nonzeros on the diagonal of the matrix.

# The elements of the matrix are stored by columns using three ``parallel""
# arrays:(link, rowSubs, values). The first element of column i stored is
# found in values(head(i)). The row subscript of the element is
# rowSubs(head(i)). The next element in the column is values(link(head(i)))
# and so on. A zero value for link marks the end of the column.

# NOTE: the elements in each column are stored in increasing
# order of row subscript. Some algorithms used in the
# package depend on this fact.

# The right hand side of the matrix equation is stored in rhs,
# and the solution (when provided or computed) is stored in the array x.
# The size of the arrays is extended as required, and their lengths
# for not generally correspond to the number of nonzeros in the matrix:
# or the number of columns in the matrix.


# The user can improve efficiency by providing an estimate of the
# number of nonzeros in the matrix -  - this is done via the optional
# keyword parameter "NNZ" in the subroutine Construct.

# Similarly, the user can improve efficiency by providing
# estimates for the number of rows and columns in the matrix via the
# optional keyword parameters "nRows" and "nCols".


module SpkProblem

using SparseArrays
using ..SpkUtilities: _BIGGY, __extend
using ..SpkGrid: Grid

"""
    Problem{IT, FT}

Type of a sparse-matrix coupled linear algebraic equations problem.

Fields 
- `nrows`: 
    * number of rows in the matrix
    * number of elements in a row permutation
- `ncols`:
    * number of columns in the matrix
    * number of elements in a column permutation

Other variables:

NOTE: a user can input `(i, j, v)`, indicating that position `(i, j)` in the
matrix contains the value `v`. Or, a user can input `(i, j)`, indicating that
position `(i, j)` is nonzero, but not specifying a value there. Hence the
distinction between "nonzeros" and "nonzero values" below.

- `nnz` - number of nonzero * values * in the matrix.
- `dnz` - number of nonzero * values * on the diagonal of the matrix.
- `nedges` - number of nonzeros in the matrix.
- `dedges` - number of nonzeros on the diagonal of the matrix.

The elements of the matrix are stored by columns using three "parallel"
arrays: (`link`, `rowsubs`, `values`). The first element of column `i` stored is
found in `values[head[i]]`. The row subscript of the element is
`rowsubs[head[i]]`. The next element in the column is `values[link[head[i]]]`
and so on. A zero value for `link` marks the end of the column.

NOTE: the elements in each column are stored in increasing order of row
subscript. Some algorithms used in the package depend on this fact.

The right hand side of the matrix equation is stored in `rhs`,
and the solution (when provided or computed) is stored in the array `x`.
The size of the arrays is extended as required, and their lengths
for not generally correspond to the number of nonzeros in the matrix:
or the number of columns in the matrix.
    

The user can improve efficiency by providing an estimate of the number of
nonzeros in the matrix -  - this is done via the optional keyword
parameter `nnz`.
    
Similarly, the user can improve efficiency by providing estimates for the number
of rows and columns in the matrix via the optional keyword parameters `nrows`
and `ncols`.
"""
mutable struct Problem{IT, FT}
    info::String
    # `lenhead` is the current length of the arrays `head` and `x`.
    lenhead::IT
    # `lenlink` is the current length of the arrays `link`, `rowsubs`, `values`.
    lenlink::IT
    # `lenrhs`  is the current length of the array `rhs`.
    lenrhs::IT
    # `lastused` is the last position in `values` and `link` that is occupied.
    lastused::IT
    head::Vector{IT}
    link::Vector{IT}
    rowsubs::Vector{IT}
    # The variable `nrows`
    # - number of rows in the matrix
    # - number of elements in a row permutation
    nrows::IT
    # The variable `ncols`
    # - number of columns in the matrix
    # - number of elements in a column permutation
    ncols::IT
    # nnz - number of nonzero * values * in the matrix.
    nnz::IT
    # dnz - number of nonzero * values * on the diagonal of the matrix.
    dnz::IT
    # nedges - number of nonzeros in the matrix.
    nedges::IT
    # dedges - number of nonzeros on the diagonal of the matrix.
    dedges::IT
    values::Vector{FT}
    rscales::Vector{FT}
    cscales::Vector{FT}
    x::Vector{FT}
    rhs::Vector{FT}
end

"""
    Problem(nrows::IT, ncols::IT, nnz::IT=2500, z::FT=0.0, info = "") where {IT, FT}

Construct a problem.
"""
function Problem(nrows::IT, ncols::IT, nnz::IT=2500, z::FT=0.0, info = "") where {IT, 
    FT}
    lenlink = nnz
    lenhead = ncols
    lenrhs = nrows

    lastused = zero(IT)
    nedges = zero(IT)
    dnz = zero(IT)
    dedges = zero(IT)
    
    head = fill(zero(IT), lenhead)
    rhs = fill(zero(FT), lenrhs)
    x = fill(zero(FT), lenhead)
    link = fill(zero(IT), lenlink)
    rowsubs = fill(zero(IT), lenlink)
    values = fill(_BIGGY(), lenlink)
    rscales = FT[]
    cscales = FT[]
    
    return Problem{IT,FT}(info, lenhead, lenlink, lenrhs, lastused, head,
        link, rowsubs, nrows, ncols, nnz, dnz, nedges, dedges, values,
        rscales, cscales, x, rhs)
end

"""
    inaij!(p::Problem{IT,FT}, rnum, cnum, aij=zero(FT)) where {IT,FT}

Input a matrix coefficient. 

The value is *added* to the existing contents.
"""
function inaij!(p::Problem{IT,FT}, rnum, cnum, aij=zero(FT)) where {IT,FT}
    if (rnum < 1 || cnum < 1)
        @warn "$(@__FILE__): invalid matrix subscripts $(rnum), $(cnum): input ignored"
        return false
    end

    if (p.lastused > p.lenlink - 1)
        p.lenlink = max(2 * p.lenlink, 3 * p.ncols)
        p.link = __extend(p.link, p.lenlink)
        p.rowsubs = __extend(p.rowsubs, p.lenlink)
        p.values = __extend(p.values, p.lenlink, _BIGGY())
    end

    p.nrows = max(rnum, p.nrows)
    p.ncols = max(p.ncols, cnum)

    if (p.ncols > p.lenhead)
        p.lenhead = 2 * p.ncols
        p.head = __extend(p.head, p.lenhead)
        p.x = __extend(p.x, p.lenhead)
    end

    if (p.nrows > p.lenrhs)
        p.lenrhs = 2 * p.nrows
        p.rhs = __extend(p.rhs, p.lenrhs)
    end

    ptr = p.head[cnum]    # ^ * check if rnum is in the row list.
    lastptr = 0
    while (ptr > 0)
        if (p.rowsubs[ptr] > rnum)
            break
        end
        if (p.rowsubs[ptr] == rnum)
            if (p.values[ptr] == _BIGGY())
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
    end                 # not there; add to column

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

"""
    inbi!(p::Problem{IT, FT}, rnum::IT, bi::FT) where {IT, FT}

Input an entry of the right-hand side vector.
"""
function inbi!(p::Problem{IT, FT}, rnum::IT, bi::FT) where {IT, FT}
    if (rnum < 1)
        @error "$(@__FILE__): Invalid rhs subscript $(rnum)."
        return false
    end

    p.nrows = max(rnum, p.nrows)
    if (p.nrows > p.lenrhs)
        p.lenrhs = 2 * p.nrows; p.rhs = __extend(p.rhs, p.lenrhs) 
    end

    p.rhs[rnum] = p.rhs[rnum] + bi 
    return true
end

"""
    insparse!(p::Problem{IT,FT}, spm) where {IT,FT}

Input sparse matrix.

Build a problem from a sparse matrix.
"""
function insparse!(p::Problem{IT,FT}, spm) where {IT,FT}
    I, J, V = findnz(spm)
    return insparse!(p, I, J, V)
end

"""
    insparse!(p::Problem{IT,FT}, I::Vector{IT}, J::Vector{IT}, V::Vector{FT}) where 
    {IT,FT}

Build a problem from a sparse matrix in the COO format.
"""
function insparse!(p::Problem{IT,FT}, I::Vector{IT}, J::Vector{IT}, V::Vector{FT}) where {IT,FT}
    for i in eachindex(I)
        if ! inaij!(p, I[i], J[i], V[i])
            return false 
        end
    end
    return true
end

"""
    outsparse(p::Problem{IT,FT})  where {IT,FT}

Output the sparse matrix.
"""
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


"""
    makegridproblem(g::Grid{IT}) where {IT}

This routine fills in a problem object using a given grid.

Input:
- `g` - the `Grid` to be used to fill a `Problem` matrix

Output:
- `p` - the Problem object to be filled
"""
function makegridproblem(g::Grid{IT}) where {IT}
    M1 = -1.0; FOUR = 4.0
    n = g.h * g.k
    p = Problem(n, n)

    for i in 1:g.h
        for j in 1:g.k
            inaij!(p, g.v[i, j], g.v[i, j], FOUR)
            if (i > 1) inaij!(p, g.v[i, j], g.v[i - 1, j], M1); end
            if (j > 1) inaij!(p, g.v[i, j], g.v[i, j - 1], M1); end
        end
    end 

    for i in 1:g.h
        for j in 1:g.k
            inaij!(p, g.v[i, j], g.v[i, j], FOUR)
            if (i>1 && j>1)   inaij!(p, g.v[i, j], g.v[i - 1, j - 1], M1); end
            if (j<g.k && i>1) inaij!(p, g.v[i, j], g.v[i - 1, j + 1], M1); end
        end
    end

    return p
end
    
"""
    makegridproblem(h::IT, k::IT) where {IT}

Construct a problem object based on a grid.

Input:
- `h` - the number of rows in the Grid
- `k` - the number of columns in the Grid

Output:
- `p` - the Problem object to be filled
"""
function makegridproblem(h::IT, k::IT) where {IT}
    g = Grid(h, k)
    return makegridproblem(g)
end

"""
    makerhs!(p::Problem, x::Vector{FT} = FT[], mtype = "T") where {FT}

This routine constructs the RHS of a problem given an `x` for the
equation `Ax = rhs`. The `x` must have the same number of elements
as the problem (represented by A above) has columns.
If `x` is not present,  a right hand side is contructed so that
the solution is 1, 2, 3, ...m.

Input Parameter:
  x - the vector in the equation ``Ax = rhs""
  mType - matrix type (optional). If the matrix is symmetric and only
            the lower or upper triangle is present, the user must let
            the routine know this by setting mType to one of:
                "L" or "l" - when only the lower triangle is present
                "U" or "u" - when only the upper triangle is present
                "T" or "t" - when either the lower or upper triangle is
                             present.
Updated Parameter:
   p - the problem for which the RHS is being constructed.
"""
function makerhs!(p::Problem, x::Vector{FT}, mtype = "T") where {FT}
    if (p.nnz == 0)
        @error "$(@__FILE__): Matrix is NULL. The rhs cannot be computed."
        return p
    end

    if (!isempty(x) )  
        p.x .= x
    else
        p.x .= FT.(1:p.ncols)
    end

    p.rhs .= 0.0
    res = deepcopy(p.rhs)

    computeresidual(p, res, p.x, mtype)

    p.rhs .= -res
    p.x .= 0.0

    return p
end


"""
    computeresidual(p::Problem, res::Vector{FT}, xin::Vector{FT} = FT[], mtype = "T") where {FT}

Compute the residual of a problem.

Given a vector `x`, this routine calculates the difference between the RHS of
the given Problem and `A*x` and places this in `res`.

Input:
- `p` - the `Problem` used to calculate `res`, using `xin`
- `xin` - the input "solution" vector used to compute the residual
- `mtype` - matrix type (optional). If the matrix is symmetric and only
            the lower or upper triangle is present, the user must let
            the routine know this by setting mType to one of:
                "L" or "l" - when only the lower triangle is present
                "U" or "u" - when only the upper triangle is present
                "T" or "t" - when either the lower or upper triangle is
                             present.

Output:
- `res` - the calculated residual
"""
function computeresidual(p::Problem, res::Vector{FT}, xin::Vector{FT} = FT[], mtype = "T") where {FT}
        # type (problem) :: p
        # real (double), dimension(:), intent(out) :: res
        # real (double) :: x(p.ncols)

#       real (doubledouble) :: t, temp, r(p.nrows), u#       
        # real (double) :: t, temp, r(p.nrows), u

        # real (double), dimension(:), optional :: xin
        # integer :: rnum, cnum, ptr, flag
        # character (len = *), optional :: mtype
        # character (len = *), parameter :: fname = "computeresidualproblem:"

    flag = 0
    if (lowercase(mtype) == "t" || lowercase(mtype) == "l" || lowercase(mtype) == "u")
        flag = 1
    else
        @error "$(@__FILE__): Invalid value for mtype, $mtype."
        return false
    end

    x = deepcopy(res)
    if (isempty(xin))
        @. x = xin
    else
        @. x = p.x
    end

    r = deepcopy(p.rhs)
    @. r = p.rhs

    for cnum in 1:p.ncols
        ptr = p.head[cnum]; t = x[cnum]
        while (ptr > 0)
            rnum = p.rowsubs[ptr];  temp = p.values[ptr]
            if (temp != _BIGGY())
                r[rnum] -= t * temp
                if (rnum != cnum && flag == 1)
                    u = x[rnum]
                    r[cnum] -= u * temp
                end
            end
            ptr = p.link[ptr]
        end
    end

    @. res = r
    return true
end

"""
    infullrhs!(p::Problem{IT,FT}, rhs)  where {IT,FT}

InRHSProblem adds a vector of values, rhs, to the current right hand
side of a problem object.

Input:
- `rhs` - the source right-hand side. It *must* be of length at least
        `p.nrows` and if it is greater than `p.nrows`, only the first
        `p.nrows` are used.

Updated:
- `p` - the problem in which rhs is to be inserted.
"""
function infullrhs!(p::Problem{IT,FT}, rhs)  where {IT,FT}
    for i in p.nrows:-1:1
        inbi!(p, i, FT(rhs[i]))
    end
    return p
end

end # module 




