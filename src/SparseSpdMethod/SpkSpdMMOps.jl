"""
This module contains the common subroutines for Matrix - Matrix
operations for both the LDLt and LU factorization algorithms.
It also contains the subroutine "LUSwap" which is only used by
the LU factorization.
"""
module SpkSpdMMops

"""
    assmb(tlen::IT, nj::IT, temp::Matrix{FT}, relcol::Vector{IT}, relind::Vector{IT}, xlnz::Vector{IT}, lnz::Vector{FT}, jlen::IT) where {IT, FT}

This routine performs an indexed assembly (i.e., scatter - add)
operation, assuming data structures used in some of our sparse
lu codes.

input parameters:
tlen - number of rows in temp.
nj - number of columns in temp.
temp - block update to be incorporated into factor storage.
relcol - relative column indices.
relind - relative indices for mapping the updates onto the target columns.
xlnz - pointers to the start of each column in the target matrix.
jlen - the length the matrix to be updated

output parameters:
lnz - contains columns modified by the update matrix.
"""
function assmb(tlen::IT, nj::IT, temp::Matrix{FT}, relcol::Vector{IT}, relind::Vector{IT}, xlnz::Vector{IT}, lnz::Vector{FT}, jlen::IT) where {IT, FT}
    @assert size(temp) == (tlen, nj)
    for j in 1:nj
        lbot = xlnz[jlen - relcol[j] + 1] - 1
        lnz[lbot - relind[1:tlen]] += temp[1:tlen, j]
    end
    return true
end

"""
    ldindx(jlen::IT, lindx::Vector{IT}, indmap::Vector{IT}) where {IT}

This routine computes the second index vector used to implement the
doubly-indirect saxpy-like loops that allow us to accumulate update columns
directly into factor storage.

     input parameters -
        jlen - length of the first column of the supernode,
                 including the diagonal entry.
        lindx - the off - diagonal row indices of the supernode,
                 i.e., the row indices of the nonzero entries
                 lying below the diagonal entry of the first
                 column of the supernode.

     output parameters -
        indmap - this index vector maps every global row index
                 of nonzero entries in the first column of the
                 supernode to its position in the index list
                 relative to the last index in the list.  more
                 precisely, it gives the distance of each index
                 from the last index in the list.

"""
function ldindx(jlen::IT, lindx::Vector{IT}, indmap::Vector{IT}) where {IT}
    indmap[lindx[1:jlen]] .= (jlen - 1):-1:0
    return true
end

# *
# *
#         igathr .... integer gather operation      ^^^^^^^
# *
# *
#
#     purpose - this routine performs a standard integer gather
#               operation.
#
#     input parameters -
#        klen - length of the list of global indices.
#        lindx - list of global indices.
#        indmap - indexed by global indices, it contains the
#                 required relative indices.
#
#     output parameters -
#        relind - list relative indices.
#
# *
function igathr(klen::IT, lindx::Vector{IT}, indmap::Vector{IT}, relind::Vector{IT}) where {IT}
    relind[1:klen] = indmap[lindx[1:klen]]
end

"""
   purpose -
       this routine performs a matrix-matrix multiply, z = z + xy,
       assuming data structures used in some of our sparse cholesky
       codes.

       matrix x has only 1 column and matrix y has only 1 row. QUESTION: Is this a rank-one update?

   input parameters -
       m       -   number of rows in x and in z.
       q       -   number of columns in y and z.
       zindxr  -   zindxr gives the list of rows in z.
       zindxc  -   zindxc gives the list of columns in z.
       x       -   contains the rows of x.
       y(*)    -   contains the columns of y.
       iz(*)   -   iz(col) points to the beginning of column col.
       relind  -   where each subscript should go in z relative to the 
                   bottom
       diag    -   the diagonal elements of the column X.  This 
                   option parameter allows for scaling of the column.

   updated parameters -
       z       -   on output, z = z + xy.

"""
function mmpyi(m::IT, q::IT, zindxr::Vector{IT}, zindxc::Vector{IT}, x::Vector{FT}, y::Vector{FT}, iz::Vector{IT}, z::Vector{FT}, relind::Vector{IT}, diag = one(eltype(x))) where {IT, FT}
    @assert length(x) == m
    @assert length(y) == q
    for k in 1:q
        t = y[k]*diag;  zlast = iz[zindxc[k]+1] - 1
        z[zlast - relind[zindxr]] .-= t .* x
    end 
    return true
end

function vswap(m, va, vb)
    for j in 1:m
        t = vb[j]
        vb[j] = va[j]
        va[j] = t
    end
end

"""
   
This routine performs row swapping inside the Upper block
supernodes of the Cholesky factorization.  This operation is
done to match to the row swapping performed on the diagonal
blocks of L.

   input parameters -
       m - number of rows in the supernode.
       n - number of columns in the supernode.
       a - the matrix to have its rows pivoted.
       lda - the real length of supernode a.
       ipvt(*) - contains the pivotal information.

   updated parameters -
       a - on output, a has its rows swapped according to
                   a[i, k] = a[ipvt[i], k] (i hope)

"""
function luswap(m::IT, n::IT, av::Vector{FT}, lda::IT, ipvt::Vector{IT}) where {IT, FT}
    # swap columns i and j of a, in-place
    function swapcols!(_m::AbstractMatrix, i, j)
        i == j && return
        cols = axes(_m,2)
        @boundscheck i in cols || throw(BoundsError(_m, (:,i)))
        @boundscheck j in cols || throw(BoundsError(_m, (:,j)))
        for k in axes(_m,1)
            @inbounds _m[k,i],_m[k,j] = _m[k,j],_m[k,i]
        end
    end
    @assert length(ipvt) == n
    @assert length(av) == m*n
    a = reshape(av, m, n)
    p = deepcopy(ipvt) 
    count = 0
    start = 0
    while count < length(p)
        ptr = start = findnext(!iszero, p, start+1)::Int
        next = p[start]
        count += 1
        while next != start
            swapcols!(a, ptr, next)
            p[ptr] = 0
            ptr = next
            next = p[next]
            count += 1
        end
        p[ptr] = 0
    end
    a
end

end 
