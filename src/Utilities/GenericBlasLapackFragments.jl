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
# Struct to allow strided reshape in the case where
# length(v)<lda*n, but length(v) is still large enough to hold all columns
# of the mxn submatrix
#
struct StridedReshape{T} <: AbstractMatrix{T}
    v::Union{Vector{T},SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}}
    lda::BlasInt
    m::BlasInt
    n::BlasInt
end

@inline idx(A::StridedReshape, i,j)= (j-1)*A.lda+i
Base.size(A::StridedReshape)=(A.m, A.n)
Base.getindex(A::StridedReshape,i,j)= @inbounds A.v[idx(A,i,j)]
Base.setindex!(A::StridedReshape,v,i,j)= @inbounds A.v[idx(A,i,j)]=v

#
# Reshape matrix with leading dimension lda>=m, taking into account the
# (for standard blas entirely legal)
# possibility that for the largest column only m elements are stored (instead of lda)



function strided_reshape(A::AbstractVector{T},lda,m,n)::Union{StridedReshape{T},SubArray{T, 2, Base.ReshapedArray{T, 2, SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}, Tuple{}}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},Base.ReshapedArray{T, 2, SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}, Tuple{}}} where T
    if lda == m
        #
        # In this case we can assume that the A buffer is large enough to hold
        # all elements of the reshaped matrix:
        #
        x=reshape(view(A,1:lda*n),m,n)
    else
        if length(A)>=lda*n
            #
            # Also, in this case we can assume that the A buffer is large enough to hold
            # all elements of the reshaped matrix:
            #
            vA=view(A,1:lda*n)
            # But we will only work with the mxn submatrix
            view(reshape(vA,lda,n),1:m,1:n)
        else
            # In the (rare) case where there is not enough
            # memory to hold the last column of reshape(vA,lda,n)
            # As the occurance may be rare, we probably can live with the current performance
            # hits.
            StridedReshape(A,lda,m,n)
        end
    end
end

#
# LU factorization adapted from generic_lufact! (https://github.com/JuliaLang/LinearAlgebra.jl/blob/main/src/lu.jl).
# Originally it is (like many other LA operators) defined for StridedMatrix which is a union and not an abstract type,
# so we cannot use that code directly. See https://github.com/JuliaLang/julia/issues/2345 for some discussion about this.
#
# Modifications:
# - Use ipiv passed instead of creating one
# - No need to return LU object 
# - Remove unused parameters - always do pivoting anyway
#
#
function ggetrf!(m,n,A::AbstractVector{FT},lda,ipiv) where FT

    minmn = min(m,n)
    idx(i,j)=(j-1)*lda+i

    @inbounds begin
        for k = 1:minmn
            # find index max
            kp = k
            if k < m #   pivot === RowMaximum() &&
                amax = abs(A[idx(k,k)])
                for i = k+1:m
                    absi = abs(A[idx(i,k)])
                    if absi > amax
                        kp = i
                        amax = absi
                    end
                end
            end
            ipiv[k] = kp
            if !iszero(A[idx(kp,k)])
                if k != kp
                    # Interchange
                    for i = 1:n
                        tmp = A[idx(k,i)]
                        A[idx(k,i)] = A[idx(kp,i)]
                        A[idx(kp,i)] = tmp
                    end
                end
                # Scale first column
                Akkinv = inv(A[idx(k,k)])
                for i = k+1:m
                    A[idx(i,k)] *= Akkinv
                end
            end
            # Update the rest
            for j = k+1:n
                for i = k+1:m
                    A[idx(i,j)] -= A[idx(i,k)]*A[idx(k,j)]
                end
            end
        end
    end
    0
end




#
# C=alpha*transA(A)*transB(B) + beta*C
#
function ggemm!(transA,transB,m,n,k,alpha,A::AbstractVector{T},lda,B::AbstractVector{T},ldb,beta,C::AbstractVector{T},ldc) where T
    oneT=one(T)
    zeroT=zero(T)
    nota= (transA=='n')
    notb= (transB=='n')


    aidx(i,j)=(j-1)*lda+i
    bidx(i,j)=(j-1)*ldb+i
    cidx(i,j)=(j-1)*ldc+i

    
    # skip the iput tests

    
    if m==0 || n==0 || ( (iszero(alpha) || k==0) && isone(beta))
        return
    end
    
    if iszero(alpha)
        if iszero(beta)
            @inbounds for i=1:n*m
                C[i]=zeroT
            end
        else
            @inbounds for i=1:n*m
                C[i]*=beta*C[i]
            end
        end
        return 
    end
    
    if notb
        if nota
            #   @Inbounds Form  C := alpha*A*B + beta*C.
            @inbounds for j = 1:n # DO 90
                if iszero(beta)
                    @inbounds for i=1:m
                        C[cidx(i,j)] = zeroT
                    end
                elseif !isone(beta)
                    @inbounds for i=1:m
                        C[cidx(i,j)] *= beta
                    end
                end
                @inbounds for l=1:k
                    temp = alpha*B[bidx(l,j)]
                    @inbounds for i=1:m
                        C[cidx(i,j)] += temp*A[aidx(i,l)]
                    end
                end
            end
        else
            #  @Inbounds Form  C := alpha*A**T*B + beta*C
            @inbounds for j=1:n
                @inbounds for i=1:m
                    temp=zeroT
                    @inbounds for l=1:k
                        temp += A[aidx(l,i)]*B[bidx(l,j)]
                    end
                    if beta==zeroT
                        @inbounds   C[cidx(i,j)] = alpha*temp
                    else
                        @inbounds  C[cidx(i,j)] = alpha*temp + beta*C[cidx(i,j)]
                    end
                end
            end
        end
    else
        if nota
            # @Inbounds Form  C := alpha*A*B**T + beta*C
            @inbounds for j=1:n
                if iszero(beta)
                    @inbounds for i=1:m
                        C[cidx(i,j)] = zeroT
                    end
                elseif !isone(beta)
                    @inbounds for i=1:m
                        C[cidx(i,j)] = beta*C[cidx(i,j)]
                    end
                end
                @inbounds for l=1:k
                    temp = alpha*B[bidx(j,l)]
                    @inbounds for i=1:m
                        C[cidx(i,j)] += temp*A[aidx(i,l)]
                    end
                end
            end
        else
            #  @Inbounds Form  C := alpha*A**T*B**T + beta*C
            @inbounds for j=1:n
                @inbounds for i=1:m
                    temp=zeroT
                    @inbounds for l=1:k
                        temp +=  A[aidx(l,i)]*B[bidx(j,l)]
                    end
                    if iszero(beta)
                        @inbounds   C[cidx(i,j)] = alpha*temp
                    else
                        @inbounds C[cidx(i,j)] = alpha*temp + beta*C[cidx(i,j)]
                    end
                end
            end
        end
    end
end



#
# Y=alpha*transA(A)*X + beta*Y
#
function ggemv!(transA,m,n,alpha,A::AbstractVector{T},lda,X,beta,Y) where T
    if m==0 || n==0
        return
    end
    
    if transA=='n'
        @inbounds for i=1:m
            Y[i]*=beta
        end
        ii0=1
        @inbounds for j=1:n # DO 60
            ii=ii0
            alphax=alpha*X[j]
            @inbounds for i=1:m # DO 50
                Y[i]+=alphax*A[ii]
                ii+=1
            end
            ii0+=lda
        end
    else
        ii0=1
        @inbounds for j=1:n # DO 120
            Y[j]*=beta
            temp=zero(T)
            ii=ii0
            for i=1:m # DO 110
                temp+=A[ii]*X[i]
                ii+=1
            end
            Y[j]+=alpha*temp
            ii0+=lda
        end
    end
    true
end





#
# Triangular solve
#
function gtrsm!(side,uplo,transA,diag, m,n,alpha,A::AbstractVector{T}, lda, B::AbstractVector{T}, ldb) where T

    if m==0 || n==0
        return
    end
    
    aidx(i,j)=(j-1)*lda+i
    bidx(i,j)=(j-1)*ldb+i
    
    oneT=one(T)
    zeroT=zero(T)
    
    lside = (side=='l')
    nounit = (diag=='n')
    upper = (uplo=='u')
    
    # skip input check
    
    if iszero(alpha)
        @inbounds for i=1:m*n
            B[i]=zeroT
        end
        return
    end
    
    if lside
        if transA== 'n'
            # @Inbounds Form  B := alpha*inv( A )*B.
            if upper
                @inbounds for j=1:n
                    if !isone(alpha)
                        @inbounds for i=1:m
                            B[bidx(i,j)] *= alpha
                        end
                    end
                    @inbounds for k=m:-1:1
                        if !iszero(B[bidx(k,j)])
                            if nounit
                                B[bidx(k,j)] /= A[aidx(k,k)]
                            end
                            @inbounds for i=1:k-1
                                B[bidx(i,j)] -= B[bidx(k,j)]*A[aidx(i,k)]
                            end
                        end
                    end
                end
            else # !upper
                @inbounds for j=1:n
                    if !isone(alpha)
                        @inbounds for i=1:m
                            B[bidx(i,j)] *= alpha
                        end
                    end
                    @inbounds for k=1:m
                        if !iszero(B[bidx(k,j)])
                            if nounit
                                B[bidx(k,j)] /= A[aidx(k,k)]
                            end
                            @inbounds for i=k+1:m
                                B[bidx(i,j)] -=  B[bidx(k,j)]*A[aidx(i,k)]
                            end
                        end
                    end
                end
            end
        else # transa=='t'
            #  @Inbounds Form  B := alpha*inv( A**T )*B.
            if upper
                @inbounds for j=1:n
                    @inbounds for i=1:m
                        temp = alpha*B[bidx(i,j)]
                        @inbounds for k=1:i-1
                            temp -=  A[aidx(k,i)]*B[bidx(k,j)]
                        end
                        B[bidx(i,j)] = temp
                        if nounit
                            temp /= A[aidx(i,i)]
                        end
                        B[bidx(i,j)]=temp
                    end
                end
            else #!upper
                @inbounds for j=1:n
                    @inbounds for i=m:-1:1
                        temp = alpha*B[bidx(i,j)]
                        @inbounds for k=i+1:m
                            temp -= A[aidx(k,i)]*B[bidx(k,j)]
                        end
                        if nounit
                            temp /= A[aidx(i,i)]
                        end
                        B[bidx(i,j)] = temp
                    end
                end
            end
        end
    else # !lside
        if transA== 'n'
            #    @Inbounds Form  B := alpha*B*inv( A ).
            if upper
                @inbounds for j=1:n
                    if !isone(alpha)
                        @inbounds for i=1:m
                            B[bidx(i,j)] = alpha*B[bidx(i,j)]
                        end
                    end
                    @inbounds for k=1:j-1
                        if !iszero(A[aidx(k,j)])
                            @inbounds for i=1:m
                                B[bidx(i,j)] -= A[aidx(k,j)]*B[bidx(i,k)]
                            end
                        end
                    end
                    if nounit
                        temp=oneT/A[aidx(j,j)]
                        @inbounds for i=1:m
                            B[bidx(i,j)] *= temp
                        end
                    end
                end
            else #!upper
                @inbounds for j=n:-1:1
                    if !isone(alpha)
                        @inbounds for i=1:m
                            B[bidx(i,j)] *= alpha
                        end
                    end
                    @inbounds for k=j+1:n
                        if !iszero(A[aidx(k,j)])
                            @inbounds for i=1:m
                                B[bidx(i,j)] -= A[aidx(k,j)]*B[bidx(i,k)]
                            end
                        end
                    end
                    if nounit
                        temp = oneT/A[aidx(j,j)]
                        @inbounds for i=1:m
                            B[bidx(i,j)] *= temp
                        end
                    end
                end
            end
        else
            # @Inbounds Form  B := alpha*B*inv( A**T ).
            if upper
                @inbounds for k=n:-1:1
                    if nounit
                        temp = oneT/A[aidx(k,k)]
                        @inbounds for i=1:m
                            B[bidx(i,k)] *= temp
                        end
                    end
                    @inbounds for j=1:k-1
                        if !iszero(A[aidx(j,k)])
                            temp = A[aidx(j,k)]
                            @inbounds for i=1:m
                                B[bidx(i,j)] -= temp*B[bidx(i,k)]
                            end
                        end
                    end
                    if !isone(alpha)
                        @inbounds for i=1:m
                            B[bidx(i,k)] *= alpha
                        end
                    end
                end
            else #!upper
                @inbounds for k=1:n
                    if nounit
                        temp = oneT/A[aidx(k,k)]
                        @inbounds for i=1:m
                            B[bidx(i,k)] *= temp
                        end
                    end
                    @inbounds for j=k+1:n
                        if !iszero(A[aidx(j,k)])
                            temp = A[aidx(j,k)]
                            @inbounds for i=1:m
                                B[bidx(i,j)] -=  temp*B[bidx(i,k)]
                            end
                        end
                    end
                    if !isone(alpha)
                        @inbounds for i=1:m
                            B[bidx(i,k)] *= alpha
                        end
                    end
                end
            end
        end
    end
end


#
# Vector entry swap according to ipiv
#
function glaswp!(a,lda,k1,k2,ipiv)
    @inbounds for i=k1:k2
        ip=ipiv[i]
        if ip!=i
            temp=a[i]
            a[i]=a[ip]
            a[ip]=temp
        end
    end
end


end
