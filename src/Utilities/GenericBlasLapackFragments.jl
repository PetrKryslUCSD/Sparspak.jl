module GenericBlasLapackFragments

#
# This module provides generic implementations of those parts of the BLAS/LAPACK
# API which are used in Sparspak.jl. 
# 
# With the exception of  ggetrf!, the implementations are rewrites of the corresponding routines
# from LAPACK 3.10.1 published on  https://netlib.org/lapack/explore-html/.
# As they are not used "in the wild", but rather in the controlled environment of this package,
# input consistency tests and bound checks have been removed in favor of performance. Any
# generalization for wider use should consider removal the @inbounds macros.
#
# They metods are  tested against
# the implementations in Sparspak.SpkSpdMMops which call into BLAS/LAPACK.


using LinearAlgebra


# As  Sparspak passes all matrix data as vectors, we create wrapper structs
# to organize 2D indexing which conforms to the usage in the BLAS routines.
# These are _not_ coforming to the abstract matrix interface as they miss
# the size() method. A proper size() method would have to take into account
# transposed and non-transposed cases etc.
# In the routines, the compiler will optimize them away, or rather place
# them on the stack as immutable objects, so this approach leaves no
# trace in the allocation statistics.
#
# The @inbounds macros in the accessor methods are the only ones in this
# package (besides of glawsp)

struct StridedReshape{Tv,Ti}
    v::Union{Vector{Tv},SubArray{Tv, 1, Vector{Tv}, Tuple{UnitRange{Ti}}, true}}
    lda::Ti
end

@inline idx(A::StridedReshape, i,j)= (j-1)*A.lda+i
Base.getindex(A::StridedReshape,i,j)= @inbounds A.v[idx(A,i,j)]
Base.setindex!(A::StridedReshape,v,i,j)= @inbounds A.v[idx(A,i,j)]=v

Base.getindex(A::StridedReshape,i)= @inbounds A.v[i]
Base.setindex!(A::StridedReshape,v,i)= @inbounds A.v[i]=v




@static if VERSION< v"1.9"
    struct RowNonZero <: LinearAlgebra.PivotingStrategy end
    lupivottype(::Type{T}) where {T} = RowMaximum()
end


@static if VERSION >= v"1.9"
    import LinearAlgebra: lupivottype
end

#
# LU factorization adapted from generic_lufact! (https://github.com/JuliaLang/LinearAlgebra.jl/blob/main/src/lu.jl)
# with support of RowNonZero pivoting for finite fields etc.
# Originally it is (like many other LA operators) defined for StridedMatrix which is a union and not an abstract type,
# so we cannot use that code directly. See https://github.com/JuliaLang/julia/issues/2345 for some discussion about this.
#
# Modifications:
# - Use ipiv passed instead of creating one
# - No need to return LU object 
#
#
function ggetrf!(m,n,a::AbstractVector{FT},lda,ipiv; pivot::Union{NoPivot,RowMaximum,RowNonZero}=lupivottype(FT)) where FT

    minmn = min(m,n)
    A=StridedReshape(a,lda)
    begin
        for k = 1:minmn
            # find index max
            kp = k
            if pivot === RowMaximum() && k < m
                amax = abs(A[k,k])
                for i = k+1:m
                    absi = abs(A[i,k])
                    if absi > amax
                        kp = i
                        amax = absi
                    end
                end
            elseif pivot === RowNonZero()
                for i = k:m
                    if !iszero(A[i,k])
                        kp = i
                        break
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
    0
end




#
# C=alpha*transA(A)*transB(B) + beta*C
#
function ggemm!(transA,transB,m,n,k,alpha,a::AbstractVector{T},lda,b::AbstractVector{T},ldb,beta,c::AbstractVector{T},ldc) where T
    oneT=one(T)
    zeroT=zero(T)
    nota= (transA=='n')
    notb= (transB=='n')

    A=StridedReshape(a,lda)
    B=StridedReshape(b,ldb)
    C=StridedReshape(c,ldc)
    
    
    # skip the iput tests
    
    
    if m==0 || n==0 || ( (iszero(alpha) || k==0) && isone(beta))
        return
    end
    
    if iszero(alpha)
        if iszero(beta)
            for i=1:n*m
                C[i]=zeroT
            end
        else
            for i=1:n*m
                C[i]*=beta*C[i]
            end
        end
        return 
    end
    
    if notb
        if nota
            #  C := alpha*A*B + beta*C.
            for j = 1:n # DO 90
                if iszero(beta)
                    for i=1:m
                        C[i,j] = zeroT
                    end
                elseif !isone(beta)
                    for i=1:m
                        C[i,j] *= beta
                    end
                end
                for l=1:k
                    temp = alpha*B[l,j]
                    for i=1:m
                        C[i,j] += temp*A[i,l]
                    end
                end
            end
        else
            #   C := alpha*A**T*B + beta*C
            for j=1:n
                for i=1:m
                    temp=zeroT
                    for l=1:k
                        temp += A[l,i]*B[l,j]
                    end
                    if beta==zeroT
                        C[i,j] = alpha*temp
                    else
                        C[i,j] = alpha*temp + beta*C[i,j]
                    end
                end
            end
        end
    else
        if nota
            #   C := alpha*A*B**T + beta*C
            for j=1:n
                if iszero(beta)
                    for i=1:m
                        C[i,j] = zeroT
                    end
                elseif !isone(beta)
                    for i=1:m
                        C[i,j] = beta*C[i,j]
                    end
                end
                for l=1:k
                    temp = alpha*B[j,l]
                    for i=1:m
                        C[i,j] += temp*A[i,l]
                    end
                end
            end
        else
            #   C := alpha*A**T*B**T + beta*C
            for j=1:n
                for i=1:m
                    temp=zeroT
                    for l=1:k
                        temp +=  A[l,i]*B[j,l]
                    end
                    if iszero(beta)
                        C[i,j] = alpha*temp
                    else
                        C[i,j] = alpha*temp + beta*C[i,j]
                    end
                end
            end
        end
    end
end



#
# Y=alpha*transA(A)*X + beta*Y
#
function ggemv!(transA,m,n,alpha,a::AbstractVector{T},lda,X,beta,Y) where T
    if m==0 || n==0
        return
    end
    
    A=StridedReshape(a,lda)
    
    if transA=='n'
        for i=1:m
            Y[i]*=beta
        end
        ii0=1
        for j=1:n # DO 60
            ii=ii0
            alphax=alpha*X[j]
            for i=1:m # DO 50
                Y[i]+=alphax*A[ii]
                ii+=1
            end
            ii0+=lda
        end
    else
        ii0=1
        for j=1:n # DO 120
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
function gtrsm!(side,uplo,transA,diag, m,n,alpha,a::AbstractVector{T}, lda, b::AbstractVector{T}, ldb) where T

    if m==0 || n==0
        return
    end
    A=StridedReshape(a,lda)
    B=StridedReshape(b,ldb)

    oneT=one(T)
    zeroT=zero(T)
    
    lside = (side=='l')
    nounit = (diag=='n')
    upper = (uplo=='u')
    
    # skip input check
    
    if iszero(alpha)
        for i=1:m*n
            B[i]=zeroT
        end
        return
    end
    
    if lside
        if transA== 'n'
            # form  B := alpha*inv( A )*B.
            if upper
                for j=1:n
                    if !isone(alpha)
                        for i=1:m
                            B[i,j] *= alpha
                        end
                    end
                    for k=m:-1:1
                        if !iszero(B[k,j])
                            if nounit
                                B[k,j] /= A[k,k]
                            end
                            for i=1:k-1
                                B[i,j] -= B[k,j]*A[i,k]
                            end
                        end
                    end
                end
            else # !upper
                for j=1:n
                    if !isone(alpha)
                        for i=1:m
                            B[i,j] *= alpha
                        end
                    end
                    for k=1:m
                        if !iszero(B[k,j])
                            if nounit
                                B[k,j] /= A[k,k]
                            end
                            for i=k+1:m
                                B[i,j] -=  B[k,j]*A[i,k]
                            end
                        end
                    end
                end
            end
        else # transa=='t'
            #  form  B := alpha*inv( A**T )*B.
            if upper
                for j=1:n
                    for i=1:m
                        temp = alpha*B[i,j]
                        for k=1:i-1
                            temp -=  A[k,i]*B[k,j]
                        end
                        B[i,j] = temp
                        if nounit
                            temp /= A[i,i]
                        end
                        B[i,j]=temp
                    end
                end
            else #!upper
                for j=1:n
                    for i=m:-1:1
                        temp = alpha*B[i,j]
                        for k=i+1:m
                            temp -= A[k,i]*B[k,j]
                        end
                        if nounit
                            temp /= A[i,i]
                        end
                        B[i,j] = temp
                    end
                end
            end
        end
    else # !lside
        if transA== 'n'
            #    form  B := alpha*B*inv( A ).
            if upper
                for j=1:n
                    if !isone(alpha)
                        for i=1:m
                            B[i,j] = alpha*B[i,j]
                        end
                    end
                    for k=1:j-1
                        if !iszero(A[k,j])
                            for i=1:m
                                B[i,j] -= A[k,j]*B[i,k]
                            end
                        end
                    end
                    if nounit
                        temp=oneT/A[j,j]
                        for i=1:m
                            B[i,j] *= temp
                        end
                    end
                end
            else #!upper
                for j=n:-1:1
                    if !isone(alpha)
                        for i=1:m
                            B[i,j] *= alpha
                        end
                    end
                    for k=j+1:n
                        if !iszero(A[k,j])
                            for i=1:m
                                B[i,j] -= A[k,j]*B[i,k]
                            end
                        end
                    end
                    if nounit
                        temp = oneT/A[j,j]
                        for i=1:m
                            B[i,j] *= temp
                        end
                    end
                end
            end
        else
            # form  B := alpha*B*inv( A**T ).
            if upper
                for k=n:-1:1
                    if nounit
                        temp = oneT/A[k,k]
                        for i=1:m
                            B[i,k] *= temp
                        end
                    end
                    for j=1:k-1
                        if !iszero(A[j,k])
                            temp = A[j,k]
                            for i=1:m
                                B[i,j] -= temp*B[i,k]
                            end
                        end
                    end
                    if !isone(alpha)
                        for i=1:m
                            B[i,k] *= alpha
                        end
                    end
                end
            else #!upper
                for k=1:n
                    if nounit
                        temp = oneT/A[k,k]
                        for i=1:m
                            B[i,k] *= temp
                        end
                    end
                    for j=k+1:n
                        if !iszero(A[j,k])
                            temp = A[j,k]
                            for i=1:m
                                B[i,j] -=  temp*B[i,k]
                            end
                        end
                    end
                    if !isone(alpha)
                        for i=1:m
                            B[i,k] *= alpha
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
