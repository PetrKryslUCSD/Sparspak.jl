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
    @views rC= reshape(C[1:m*n],m,n)
    @views rA= transA=='n' ? reshape(A[1:k*m],m,k) :  transpose(reshape(A[1:k*m],k,m))
    @views rB= transB=='n' ? reshape(B[1:k*n],k,n) :  transpose(reshape(B[1:k*n],n,k))
    mul!(rC,rA,rB,alpha, beta)
    true
end


#
# Y=alpha*transA(A)*X + beta*Y
#
function ggemv!(transA,m,n,alpha,A,lda,X,beta,Y)
    if transA=='n'
        @views mul!(Y[1:m],reshape(A[1:m*n],m,n),X[1:n],alpha, beta)
    else
        @views mul!(Y[1:n],transpose(reshape(A[1:m*n],m,n)),X[1:m],alpha, beta)
    end
    true
end


#
# In-place LU factorization of A
#
function ggetrf!(m,n,A::AbstractVector{FT},lda,ipiv) where FT
    @views ipiv[1:n].=lu!(reshape(A[1:m*n],m,n)).p
    return 0
end



#
# Triangular solve
#
function gtrsm!(side,uplo,transA,diag, m,n,alpha,A, lda, B, ldb)
    
    k= side=='l' ? m : n
    if transA=='t'
        @views kA=transpose(reshape(A[1:k*k],k,k))
    else
        @views kA=reshape(A[1:k*k],k,k)
    end
    if diag=='n'
        if uplo=='u'
            tA=UpperTriangular(kA)
        elseif uplo=='l'
            tA=LowerTriangular(kA)
        end
    elseif diag == 'u'
        if uplo=='u'
            tA=UnitUpperTriangular(kA)
        elseif uplo=='l'
            tA=UnitLowerTriangular(kA)
        end
    end

    @views rB=reshape(B[1:m*n],m,n)
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
