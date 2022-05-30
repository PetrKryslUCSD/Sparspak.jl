"""
This module contains the common subroutines for Matrix - Matrix
operations for both the LDLt and LU factorization algorithms.
It also contains the subroutine "LUSwap" which is only used by
the LU factorization.
"""
module SpkSpdMMops

using LinearAlgebra
using LinearAlgebra.LAPACK: getrf!
using Libdl

# import LinearAlgebra, OpenBLAS64_jll
# LinearAlgebra.BLAS.lbt_forward(OpenBLAS64_jll.libopenblas_path)

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
function ldindx(jlen::IT, lindx::SubArray{IT, 1, Vector{IT}, Tuple{UnitRange{IT}}, true}, indmap::Vector{IT}) where {IT}
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
function mmpyi(m::IT, q::IT, 
    zindxr::SubArray{IT, 1, Vector{IT}, Tuple{UnitRange{IT}}, true}, 
    zindxc::SubArray{IT, 1, Vector{IT}, Tuple{UnitRange{IT}}, true}, 
    x::SubArray{FT, 1, Vector{FT}, Tuple{UnitRange{IT}}, true}, 
    y::SubArray{FT, 1, Vector{FT}, Tuple{UnitRange{IT}}, true}, 
    iz::Vector{IT},
    z::Vector{FT},
    relind::Vector{IT}, diag = one(eltype(x))) where {IT, FT}
    @assert length(x) >= m
    @assert length(y) >= q
    for k in 1:q
        t = y[k]*diag;  zlast = iz[zindxc[k]+1] - 1
        for j in 1:m
            zi = relind[zindxr[j]]
            z[zlast - zi] -= t * x[j]
        end
    end 
    return true
end

function vswap(m, va, vb)
    @show m, va, vb
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
# function luswap(m::IT, n::IT, av::SubArray{FT, 1, Vector{FT}, Tuple{UnitRange{IT}}, true}, lda::IT, ipvt::SubArray{IT, 1, Vector{IT}, Tuple{UnitRange{IT}}, true}) where {IT, FT}
#     # swap columns i and j of a, in-place
#     function swapcols!(_m::AbstractMatrix, i, j)
#         i == j && return
#         cols = axes(_m,2)
#         @boundscheck i in cols || throw(BoundsError(_m, (:,i)))
#         @boundscheck j in cols || throw(BoundsError(_m, (:,j)))
#         for k in axes(_m,1)
#             @inbounds _m[k,i],_m[k,j] = _m[k,j],_m[k,i]
#         end
#     end
#     @assert length(ipvt) >= n
#     @assert length(av) >= m*n
#     @show av, m, n
#     a = reshape(view(av, 1:(m*n)), m, n)
#     @show p = deepcopy(ipvt) 
#     count = 0
#     start = 0
#     while count < length(p)
#         ptr = start = findnext(!iszero, p, start+1)::Int
#         next = p[start]
#         count += 1
#         while next != start
#             swapcols!(a, ptr, next)
#             p[ptr] = 0
#             ptr = next
#             next = p[next]
#             count += 1
#         end
#         p[ptr] = 0
#     end
#     a
# end
# luswap(jlen - nj, nj, view(unz, jupnt:length(unz)), jlen - nj, view(ipvt, fj:length(ipvt)))
function luswap(m::IT, n::IT, a::SubArray{FT, 1, Vector{FT}, Tuple{UnitRange{IT}}, true}, lda::IT, ipvt::SubArray{IT, 1, Vector{IT}, Tuple{UnitRange{IT}}, true}) where {IT, FT}
    @show m, n, length(a), lda, ipvt
    for k in 1:n
        i = ipvt[k]
        ks = (k - 1)*lda + 1
        is = (i - 1)*lda + 1
        vswap(m, view(a, ks:ks+m-1), view(a, is:is+m-1))
    end
end

const __BLAS_LIB = Ref{Ptr{Nothing}}()

function __init__()
    __BLAS_LIB[] = dlopen(LinearAlgebra.BLAS.libblastrampoline)
end

# dgemm("n", "t", jlen, nj, nk, -one(FT), lnz[klpnt], ksuplen, unz[kupnt], ksuplen - nk, one, lnz[jlpnt], jlen)
function dgemm!(transA::AbstractChar, transB::AbstractChar, m::IT, n::IT, k::IT,
    alpha::FT,
    A::AbstractVecOrMat{FT}, lda::IT,
    B::AbstractVecOrMat{FT}, ldb::IT,
    beta::FT,
    C::AbstractVecOrMat{FT}, ldc::IT) where {IT, FT}
    __DGEMM_PTR = dlsym(__BLAS_LIB[], LinearAlgebra.BLAS.@blasfunc(dgemm_))
    ccall(__DGEMM_PTR, Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt},
            Ref{LinearAlgebra.BLAS.BlasInt}, Ref{FT}, Ptr{FT}, Ref{LinearAlgebra.BLAS.BlasInt},
            Ptr{FT}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{FT}, Ptr{FT},
            Ref{LinearAlgebra.BLAS.BlasInt}, Clong, Clong),
        transA, transB, m, n, k, alpha, A, lda,
        B, ldb, beta, C, ldc, 1, 1)
    C
end

# SUBROUTINE DGETRF( M, N, A, LDA, IPIV, INFO )
# *     .. Scalar Arguments ..
#       INTEGER            INFO, LDA, M, N
# *     .. Array Arguments ..
#       INTEGER            IPIV( * )
#       DOUBLE PRECISION   A( LDA, * )
function dgetrf!(m::IT, n::IT, A::SubArray{FT, 1, Vector{FT}, Tuple{UnitRange{IT}}, true}, lda::IT, ipiv::SubArray{IT, 1, Vector{IT}, Tuple{UnitRange{IT}}, true}) where {IT, FT}
    info = Ref{LinearAlgebra.BLAS.BlasInt}()
    __DGETRF_PTR = dlsym(__BLAS_LIB[], LinearAlgebra.BLAS.@blasfunc(dgetrf_))
    ccall(__DGETRF_PTR, Cvoid,
        (Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{FT},
            Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{LinearAlgebra.BLAS.BlasInt}, Ptr{LinearAlgebra.BLAS.BlasInt}),
        m, n, A, lda, ipiv, info)
    @show info[]
    return info[] #Error code is stored in LU factorization type
end

#       SUBROUTINE DTRSM(SIDE,UPLO,TRANSA,DIAG,M,N,ALPHA,A,LDA,B,LDB)
# *     .. Scalar Arguments ..
#       DOUBLE PRECISION ALPHA
#       INTEGER LDA,LDB,M,N
#       CHARACTER DIAG,SIDE,TRANSA,UPLO
# *     .. Array Arguments ..
#       DOUBLE PRECISION A(LDA,*),B(LDB,*)
# dtrsm("r", "u", "n", "n", jlen - nj, nj, one(FT), lnz[jlpnt], jlen, lnz[jlpnt + nj], jlen)
function dtrsm!(side::AbstractChar, uplo::AbstractChar, transa::AbstractChar, diag::AbstractChar, m::IT, n::IT, alpha::FT, A::SubArray{FT, 1, Vector{FT}, Tuple{UnitRange{IT}}, true}, lda::IT, B::SubArray{FT, 1, Vector{FT}, Tuple{UnitRange{IT}}, true}, ldb::IT) where {IT, FT}
    __DTRSM_PTR = dlsym(__BLAS_LIB[], LinearAlgebra.BLAS.@blasfunc(dtrsm_))
    ccall(__DTRSM_PTR, Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
            Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{FT}, Ptr{FT},
            Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{FT}, Ref{LinearAlgebra.BLAS.BlasInt},
            Clong, Clong, Clong, Clong),
        side, uplo, transa, diag,
        m, n, alpha, A, lda, B, ldb,
        1, 1, 1, 1)
    B
end

end 
