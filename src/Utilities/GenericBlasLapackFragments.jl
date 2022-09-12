module GenericBlasLapackFragments
#
# This module provides generic implementations of those parts of the BLAS/LAPACK
# API which are used in Sparspak.jl. 
# 

# The ggemm!  etc functions are  tested against
# the implementations in Sparspak.SpkSpdMMops which call into BLAS/LAPACK.


using LinearAlgebra
using LinearAlgebra:BlasInt

#
# Reshape matrix with leading dimension lda>=m
#
function xstrided_reshape(A,lda,m,n)
    vA=view(A,1:lda*n)
    if lda == m
        reshape(vA,m,n)
    else
        view(reshape(vA,lda,n),1:m,1:n)
    end
end

struct StridedReshape{T} <: AbstractMatrix{T}
    v::Union{Vector{T},SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}}
    lda::Int
    m::Int
    n::Int
end

idx(A::StridedReshape, i,j)= (j-1)*A.lda+i
Base.size(A::StridedReshape)=(A.m, A.n)
Base.getindex(A::StridedReshape,i,j)= @inbounds A.v[idx(A,i,j)]
Base.setindex!(A::StridedReshape,v,i,j)= @inbounds A.v[idx(A,i,j)]=v

strided_reshape(A,lda,m,n)=StridedReshape(A,lda,m,n)

#
# LU factorization copied from LinearAlgebra.jl  - originally it is (like many other operators
# defined for StridedMatrix which is a union and not an abstract type
# Modifications: use ipiv passed, no need to create LU object.
function glu!(A, ipiv, pivot::Union{LinearAlgebra.RowMaximum,LinearAlgebra.NoPivot} = RowMaximum(), check::Bool = true)
    # Extract values
    m, n = size(A)
    minmn = min(m,n)
    
    @inbounds begin
        for k = 1:minmn
            # find index max
            kp = k
            if pivot === RowMaximum() && k < m
                amax = abs(A[k, k])
                for i = k+1:m
                    absi = abs(A[i,k])
                    if absi > amax
                        kp = i
                        amax = absi
                    end
                end
            end
            ipiv[k] = kp
            if !iszero(A[kp,k])
                if k != kp
                    # Interchange
                    for i = 1:n
                        tmp = A[k,i]
                        A[k,i] = A[kp,i]
                        A[kp,i] = tmp
                    end
                end
                # Scale first column
                Akkinv = inv(A[k,k])
                for i = k+1:m
                    A[i,k] *= Akkinv
                end
            end
            # Update the rest
            for j = k+1:n
                for i = k+1:m
                    A[i,j] -= A[i,k]*A[k,j]
                end
            end
        end
    end
#    check && checknonsingular(info, pivot)
end




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

#
# In-place LU factorization of A
#
function ggetrf!(m,n,A::AbstractVector{FT},lda,ipiv) where FT
    glu!(strided_reshape(A,lda,m,n),ipiv)
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

    rA=strided_reshape(A,lda,k,k)

    if diag=='n'
        if uplo=='u'
            tA=UpperTriangular(rA)
        else
            tA=LowerTriangular(rA)
        end
    else
        if uplo=='u'
            tA=UnitUpperTriangular(rA)
        else
            tA=UnitLowerTriangular(rA)
        end
    end
    
    
    if transA=='t'
        kA=transpose(tA)
    else
        kA=tA
    end
    
    
    rB=strided_reshape(B,ldb,m,n)
    
    if  side=='l'
        rB.=kA\(alpha*rB)
    else
        rB.=alpha*rB/kA
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
