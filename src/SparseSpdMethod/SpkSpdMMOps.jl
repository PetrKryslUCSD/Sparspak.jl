"""
This module contains the common subroutines for Matrix - Matrix
operations for both the LDLt and LU factorization algorithms.
It also contains the subroutine "LUSwap" which is only used by
the LU factorization.
"""
module SpkSpdMMops

using LinearAlgebra
using LinearAlgebra.LAPACK: getrf!



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

#      DGEMM  performs one of the matrix-matrix operations

#         C := alpha*op( A )*op( B ) + beta*C,

#      where  op( X ) is one of

#         op( X ) = X   or   op( X ) = X**T,

#      alpha and beta are scalars, and A, B and C are matrices, with op( A )
#      an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

# Parameters
#     [in]    TRANSA  

#               TRANSA is CHARACTER*1
#                On entry, TRANSA specifies the form of op( A ) to be used in
#                the matrix multiplication as follows:

#                   TRANSA = 'N' or 'n',  op( A ) = A.

#                   TRANSA = 'T' or 't',  op( A ) = A**T.

#                   TRANSA = 'C' or 'c',  op( A ) = A**T.

#     [in]    TRANSB  

#               TRANSB is CHARACTER*1
#                On entry, TRANSB specifies the form of op( B ) to be used in
#                the matrix multiplication as follows:

#                   TRANSB = 'N' or 'n',  op( B ) = B.

#                   TRANSB = 'T' or 't',  op( B ) = B**T.

#                   TRANSB = 'C' or 'c',  op( B ) = B**T.

#     [in]    M   

#               M is INTEGER
#                On entry,  M  specifies  the number  of rows  of the  matrix
#                op( A )  and of the  matrix  C.  M  must  be at least  zero.

#     [in]    N   

#               N is INTEGER
#                On entry,  N  specifies the number  of columns of the matrix
#                op( B ) and the number of columns of the matrix C. N must be
#                at least zero.

#     [in]    K   

#               K is INTEGER
#                On entry,  K  specifies  the number of columns of the matrix
#                op( A ) and the number of rows of the matrix op( B ). K must
#                be at least  zero.

#     [in]    ALPHA   

#               ALPHA is DOUBLE PRECISION.
#                On entry, ALPHA specifies the scalar alpha.

#     [in]    A   

#               A is DOUBLE PRECISION array, dimension ( LDA, ka ), where ka is
#                k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
#                Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
#                part of the array  A  must contain the matrix  A,  otherwise
#                the leading  k by m  part of the array  A  must contain  the
#                matrix A.

#     [in]    LDA 

#               LDA is INTEGER
#                On entry, LDA specifies the first dimension of A as declared
#                in the calling (sub) program. When  TRANSA = 'N' or 'n' then
#                LDA must be at least  max( 1, m ), otherwise  LDA must be at
#                least  max( 1, k ).

#     [in]    B   

#               B is DOUBLE PRECISION array, dimension ( LDB, kb ), where kb is
#                n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
#                Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
#                part of the array  B  must contain the matrix  B,  otherwise
#                the leading  n by k  part of the array  B  must contain  the
#                matrix B.

#     [in]    LDB 

#               LDB is INTEGER
#                On entry, LDB specifies the first dimension of B as declared
#                in the calling (sub) program. When  TRANSB = 'N' or 'n' then
#                LDB must be at least  max( 1, k ), otherwise  LDB must be at
#                least  max( 1, n ).

#     [in]    BETA    

#               BETA is DOUBLE PRECISION.
#                On entry,  BETA  specifies the scalar  beta.  When  BETA  is
#                supplied as zero then C need not be set on input.

#     [in,out]    C   

#               C is DOUBLE PRECISION array, dimension ( LDC, N )
#                Before entry, the leading  m by n  part of the array  C must
#                contain the matrix  C,  except when  beta  is zero, in which
#                case C need not be set on entry.
#                On exit, the array  C  is overwritten by the  m by n  matrix
#                ( alpha*op( A )*op( B ) + beta*C ).

#     [in]    LDC 

#               LDC is INTEGER
#                On entry, LDC specifies the first dimension of C as declared
#                in  the  calling  (sub)  program.   LDC  must  be  at  least
#                max( 1, m ).

# dgemm("n", "t", jlen, nj, nk, -one(FT), lnz[klpnt], ksuplen, unz[kupnt], ksuplen - nk, one, lnz[jlpnt], jlen)
function dgemm!(transA::AbstractChar, transB::AbstractChar, m::IT, n::IT, k::IT,
    alpha::FT,
    A::AbstractVecOrMat{FT}, lda::IT,
    B::AbstractVecOrMat{FT}, ldb::IT,
    beta::FT,
    C::AbstractVecOrMat{FT}, ldc::IT) where {IT, FT}
    ccall((LinearAlgebra.BLAS.@blasfunc(:dgemm_), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
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
    A = rand(7, 7)
    @show A, ipiv, info = getrf!(A)
    info = Ref{LinearAlgebra.BLAS.BlasInt}()
    ccall((LinearAlgebra.LAPACK.@blasfunc(:dgetrf_), LinearAlgebra.LAPACK.libblastrampoline), Cvoid,
        (Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{FT},
            Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{LinearAlgebra.BLAS.BlasInt}, Ptr{LinearAlgebra.BLAS.BlasInt}),
        m, n, A, lda, ipiv, info)
    chkargsok(info[])
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
function dtrsm!(side::AbstractChar, uplo::AbstractChar, transa::AbstractChar, diag::AbstractChar, m::IT, n::IT, alpha::FT, A::AbstractMatrix{FT}, lda::IT, B::AbstractMatrix{FT}, ldb::IT) where {IT, FT}
    if k != (side == 'L' ? m : n)
        throw(DimensionMismatch("size of A is ($k,$k), size of B is ($m,$n), side is $side, and transa='$transa'"))
    end
    ccall((LinearAlgebra.BLAS.@blasfunc(:dtrsm_), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
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
