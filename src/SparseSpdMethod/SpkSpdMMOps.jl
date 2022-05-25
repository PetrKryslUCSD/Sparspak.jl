module SpkSpdMMops
"""

* This module contains the common subroutines for Matrix - Matrix
* operations for both the LDLt and LU factorization algorithms.
* It also contains the subroutine "LUSwap" which is only used by
* the LU factorization.


*     assmb .... indexed assembly operation     ^^^^^^


*
*   purpose:
*       this routine performs an indexed assembly (i.e., scatter - add)
*       operation, assuming data structures used in some of our sparse
*       lu codes.
*
*   input parameters:
*       tlen - number of rows in temp.
*       nj - number of columns in temp.
*       temp - block update to be incorporated into factor
*                           storage.
*       relcol - relative column indices.
*       relind - relative indices for mapping the updates
*                           onto the target columns.
*       xlnz - pointers to the start of each column in the
*                           target matrix.
*       jlen - the length the matrix to be updated
*
*   output parameters:
*       lnz - contains columns modified by the update
*                           matrix.
*
"""
function assmb(tlen, nj, temp, relcol, relind, xlnz, lnz, jlen)
# integer :: jlen, tlen, nj, j, lbot
# integer :: xlnz(*), relcol(*), relind(*)
# real(double) :: lnz(*), temp(tlen, nj)

    for j = 1: nj
        lbot = xlnz[jlen - relcol[j] + 1] - 1
        lnz[lbot - relind[1:tlen]] += temp[1:tlen, j]
    end
    return true
end

#
#     purpose - this routine computes the second index vector
#               used to implement the doubly - indirect saxpy - like
#               loops that allow us to accumulate update
#               columns directly into factor storage.
#
#     input parameters -
#        jlen - length of the first column of the supernode,
#                 including the diagonal entry.
#        lindx - the off - diagonal row indices of the supernode,
#                 i.e., the row indices of the nonzero entries
#                 lying below the diagonal entry of the first
#                 column of the supernode.
#
#     output parameters -
#        indmap - this index vector maps every global row index
#                 of nonzero entries in the first column of the
#                 supernode to its position in the index list
#                 relative to the last index in the list.  more
#                 precisely, it gives the distance of each index
#                 from the last index in the list.
#
# *
function ldindx(jlen, lindx, indmap)
# integer :: j, jlen, lindx(*), indmap(*)

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
function igathr(klen, lindx, indmap, relind)
# 
      # integer :: klen, indmap(*), lindx(*), relind(*)

    relind[1:klen] = indmap[lindx[1:klen]]
end

"""
!   purpose -
!       this routine performs a matrix-matrix multiply, z = z + xy,
!       assuming data structures used in some of our sparse cholesky
!       codes.
!
!       matrix x has only 1 column and matrix y has only 1 row.
!
!   input parameters -
!       m       -   number of rows in x and in z.
!       q       -   number of columns in y and z.
!       zindxr  -   zindxr gives the list of rows in z.
!       zindxc  -   zindxc gives the list of columns in z.
!       x       -   contains the rows of x.
!       y(*)    -   contains the columns of y.
!       iz(*)   -   iz(col) points to the beginning of column col.
!       relind  -   where each subscript should go in z relative to the 
!                   bottom
!       diag    -   the diagonal elements of the column X.  This 
!                   option parameter allows for scaling of the column.
!
!   updated parameters -
!       z       -   on output, z = z + xy.
!
"""
function mmpyi(m, q, zindxr, zindxc, x, y, iz, z, relind, diag = one(eltype(x)))

# integer :: m, q, k, zlast
# integer :: iz(*), relind(*), zindxr(*), zindxc(*)
# real(double) :: x(*), y(*), z(*), t, d
# real(double), optional :: diag

    for k in 1:q
        t = y[k]*diag;  zlast = iz[zindxc[k]+1] - 1
        z[zlast - relind[zindxr[1:m]]] -= t*x[1:m]
    end 
    return true
end

#
# *    luswap  .... row swapping     ^^^^^^^^^ *
#
#
#
#   purpose -
#       this routine performs row swapping inside the Upper block
#       supernodes of the cholesky factorization.  This operation is
#       done to match to the row swapping performed on the diagonal
#       blocks of L.
#
#   input parameters -
#       m - number of rows in the supernode.
#       n - number of columns in the supernode.
#       a - the matrix to have its rows pivoted.
#       lda - the real length of supernode a.
#       ipvt(*) - contains the pivotal information.
#
#   updated parameters -
#       a - on output, a has it"s rows swapped according to
#                   a[i, k] = a[ipvt[i], k] (i hope)
#
"""
integer :: m, n, lda
        integer :: ipvt(n)
        real(double) ::  a(lda, n)
        integer :: i, k
"""
function luswap(m, n, a, lda, ipvt)
    for k in 1:n
        i = ipvt[k]
        dswap(m, a(1, k), 1, a(1, i), 1)
    end
    return
end

end 
