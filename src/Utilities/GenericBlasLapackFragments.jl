module GenericBlasLapackFragments
#
# This module provides generic implementations of those parts of the BLAS/LAPACK
# API which are used in Sparspak.jl. 
# 

# The ggemm!  etc functions are  tested against
# the implementations in Sparspak.SpkSpdMMops which call into BLAS/LAPACK.


using LinearAlgebra

strided_reshape(A,lda,m,n)= lda == m ? reshape(view(A,1:m*n),m,n) : view(reshape(view(A,1:lda*n),lda,n),1:m,1:n)


#
# C=alpha*transA(A)*transB(B) + beta*C
#
function ggemm!(transA,transB,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
    rA= transA=='n' ? strided_reshape(A,lda,m,k) :  transpose(strided_reshape(A,lda,k,m))
    rB= transB=='n' ? strided_reshape(B,ldb,k,n) :  transpose(strided_reshape(B,ldb,n,k))
    rC= strided_reshape(C,ldc,m,n)
    mul!(rC,rA,rB,alpha, beta)
    true
end


#
# Y=alpha*transA(A)*X + beta*Y
#
function ggemv!(transA,m,n,alpha,A,lda,X,beta,Y)
    if m==0 || n==0
        return
    end
    rA=strided_reshape(A,lda,m,n)
    if transA=='n'
        @views mul!(Y[1:m],rA,X[1:n],alpha, beta)
    else
        @views mul!(Y[1:n],transpose(rA),X[1:m],alpha, beta)
    end
    true
end

function xggemv!(transA,m,n,alpha,A,lda,X,beta,Y)
    if m==0 || n==0
        return
    end
    @show typeof(A)
    @show isa(A,AbstractVector{eltype(A)})
#    a=reshape(A[1:m*n],m,n)
    for i=1:length(Y)
        Y[i]*=beta
    end
    if transA=='n'
        for j=1:n # DO 60
            for i=1:m # DO 50
                Y[i]+=alpha*X[j]*A[(j-1)*lda+i]
            end
        end
    else
        for j=1:n # DO 120
            temp=zero(eltype(A))
            for i=1:m # DO 110
                temp+=A[(j-1)*lda+i]*X[i]
            end
            Y[j]+=alpha*temp
        end
    end
    true
end

#
# In-place LU factorization of A
#
function ggetrf!(m,n,A::AbstractVector{FT},lda,ipiv) where FT
    rA=strided_reshape(A,lda,m,n)
    @views ipiv[1:n].=lu!(rA).p
    return 0
end



#
# Triangular solve
#
function gtrsm!(side,uplo,transA,diag, m,n,alpha,A, lda, B, ldb)
    if m==0 || n==0
        return
    end
    
    k= side=='l' ? m : n
    if transA=='t'
        kA=transpose(strided_reshape(A,lda,k,k))
    else
        kA=strided_reshape(A,lda,k,k)
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
    rB=strided_reshape(B,ldb,m,n)
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
