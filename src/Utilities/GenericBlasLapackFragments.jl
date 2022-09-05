module GenericBlasLapackFragments
#
# This module provides generic implementations of those parts of the BLAS/LAPACK
# API which are used in Sparspak.jl. 
# 

# The ggemm!  etc functions are  tested against
# the implementations in Sparspak.SpkSpdMMops which call into BLAS/LAPACK.


using LinearAlgebra

#
# C=alpha*transA(A)*transB(B) + beta*C
#
function ggemm!(transA,transB,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    rC= reshape(C,m,n)
    rA= transA=='n' ? reshape(A,m,k) :  transpose(reshape(A,k,m))
    rB= transB=='n' ? reshape(B,k,n) :  transpose(reshape(B,n,k))
    mul!(rC,rA,rB,alpha, beta)
    true
end


#
# Y=alpha*transA(A)*X + beta*Y
#
function ggemv!(transA,m,n,alpha,A,lda,X,beta,Y)
    rA= transA=='n' ? reshape(A,m,n) :  transpose(reshape(A,m,n))
    mul!(Y,rA,X,alpha, beta)
    true
end


#
# In-place LU factorization of A
#
function ggetrf!(m,n,A,lda,ipiv)
    ipiv.=lu!(reshape(A,m,n)).p
end



#
# Triangular solve 
#
function gtrsm!(side,uplo,transa,diag, m,n,alpha,A, lda, B, ldb)
    if diag!='n'
        error("generic *trsm for unit tridiagonal matrices not implemented")
    end
    if transa!='n'
        error("generic *trsm for transposed matrices not implemented")
    end
    
    k= side=='l' ? m : n

    if uplo=='u'
        tA=UpperTriangular(reshape(A,k,k))
    elseif uplo=='l'
        tA=LowerTriangular(reshape(A,k,k))
    end
    
    rB=reshape(B,m,n)
    if  side=='l'
        rB.=tA\(alpha*rB)
    elseif side == 'r'
        transpose(rB).=transpose(tA)\transpose(alpha*rB)
    end
end



#
# Vector entry swap according to ipiv
#
function glaswp!(a,lda,k1,k2,ipiv)
    for i=k1:k2
        ip=ipiv[i]
        if ip!=i
            temp=a[i]
            a[i]=a[ip]
            a[ip]=temp
        end
    end
end


end
