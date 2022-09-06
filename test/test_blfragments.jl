module test_blfragments
using Test
using LinearAlgebra
using Random
using LinearAlgebra.BLAS: BlasInt
using Sparspak.SpkSpdMMops:  dgemm!,dgemv!,dgetrf!,dtrsm!,dlaswp!
using Sparspak.GenericBlasLapackFragments:  ggemm!,ggemv!,ggetrf!,gtrsm!,glaswp!, strided_reshape


using ForwardDiff
using MultiFloats


#
# Random number generation for duals
#
Random.rand(rng::AbstractRNG, ::Random.SamplerType{ForwardDiff.Dual{T,V,N}}) where {T,V,N} = ForwardDiff.Dual{T,V,N}(rand(rng,T))

#
# Map eps for duals and complex to underlying FP type
#
Base.eps(::Type{ForwardDiff.Dual{T,V,N}}) where {T,V,N}= Base.eps(T)
Base.eps(::Type{Complex{T}}) where {T}= Base.eps(T)


#
# Conversion to Standard FP from dual etc.
#
f64(V::AbstractVector{T}) where T=[Float64(ForwardDiff.value(V[i])) for i=1:length(V)]
f64(V::AbstractMatrix{T}) where T=[Float64(ForwardDiff.value(V[i,j])) for i=1:size(V,1), j=1:size(V,2)]
f64(x::ForwardDiff.Dual{T}) where T=Float64(ForwardDiff.value(x))
f64(x)=Float64(x)

#
# In the complex case, we keep the  complex numbers
#
f64(V::AbstractVector{Complex{T}}) where T =[Complex{Float64}(ForwardDiff.value(V[i])) for i=1:length(V)]
f64(V::AbstractMatrix{Complex{T}}) where T =[Complex{Float64}(ForwardDiff.value(V[i,j])) for i=1:size(V,1), j=1:size(V,2)]
f64(x::Complex{T}) where T = Complex{Float64}(x)


#
# Verify f64 conversion
#
function tf64()
    for n in rand(1:50,15)
        for m in rand(1:50, 15)
            A=rand(n,m)
            v=rand(m)
            @assert A==f64(A)
            @assert v==f64(v)
        end
    end
    true
end


#
# Test ggemm! for a couple of random data,
# compare timing between generic and blas  implementations
#
function tgemm(T=Float64)
    tblas=0.0
    tgnrc=0.0
    ldx()=rand(0:5)
    for m in rand(1:50,15)
        for n in rand(1:50,15)
            for k in rand(1:50,15)
                for transA in ['n','t']
                    for transB in ['n','t']

                        if transA=='n'
                            lda= ldx()+m
                            A=rand(T,lda*k)
                        else
                            lda= ldx()+k
                            A=rand(T,lda*m)
                        end
                        
                        if transB=='n'
                            ldb= ldx()+k
                            B=rand(T,ldb*n)
                        else
                            ldb= ldx()+n
                            B=rand(T,ldb*k)
                        end

                        ldc=ldx()+m
                        C=rand(T,ldc*n)

                        α=-one(T)
                        β=one(T)
                        
                        
                        A64=f64(A)
                        C64=f64(C)
                        B64=f64(B)

                        α64=f64(α)
                        β64=f64(β)
                        
                        
                        
                        tblas+=@elapsed dgemm!(transA,transB, m,n,k, α64 ,A64, lda ,B64,ldb, β64,C64,ldc)
                        tgnrc+=@elapsed ggemm!(transA,transB, m,n,k, α ,A, lda ,B,ldb, β,C,ldc)
                        if ! isapprox(f64(C),C64,rtol=10*max(eps(T),eps(Float64)))
                            error("error in $transA-$transB ($n, $m, $k)")
                        end
                    end
                end
            end
        end
    end
    @info "gemm:  tgnrc/tblas=$(tgnrc/tblas)"
    true
end


#
# Test ggemv! for a couple of random data,
# compare timing between generic and blas  implementations
#
function tgemv(T=Float64)
    tblas=0.0
    tgnrc=0.0
    ldx()=rand(0:5)
    for m in rand(1:50,15)
        for n in rand(1:50,15)
            for transA in ['n','t']
                
                if transA=='n'
                    lda= ldx()+m
                    A=rand(T,lda*n)
                else
                    lda= ldx()+n
                    A=rand(T,lda*m)
                end
   
                α=rand(T)
                β=rand(T)
                
                A64=f64(A)
                   
                α64=f64(α)
                β64=f64(β)
                
                if transA=='n'
                    X=rand(T,n)
                    Y=rand(T,m)
                else
                    X=rand(T,m)
                    Y=rand(T,n)
                end
                X64=f64(X)
                Y64=f64(Y)
                tblas+=@elapsed dgemv!(transA, m,n, α64 ,A64, m ,X64, β64,Y64)
                tgnrc+=@elapsed ggemv!(transA, m,n, α ,A, m ,X, β,Y)
                if ! isapprox(f64(Y),Y64,rtol=10*max(eps(T),eps(Float64)))
                    error("error in $transA ($n, $m)")
                end
            end
        end
    end
    @info "gemv:  tgnrc/tblas=$(tgnrc/tblas)"
    true
end




#
# Test ggetrf! for a couple of random data,
# compare timing between generic and blas  implementations
#
function tgetrf(T=Float64)
    tblas=0.0
    tgnrc=0.0
    ldx()=rand(0:5)
    for n in rand(1:50,25)
        m=n
        lda=ldx()+m
        A=-rand(T,lda*n)

        rA=strided_reshape(A,lda,m,m)
        for i=1:n
            rA[i,i]=2.0*max(m,n)  
        end
        
        ipiv=zeros(BlasInt,min(m,n))
        ipiv64=zeros(BlasInt,min(m,n))
        
        A64=f64(A)

        tblas+=@elapsed Alu64=dgetrf!(m,n,A64,lda,ipiv64)
        tgnrc+=@elapsed Alu=ggetrf!(m,n,A,lda,ipiv)
        if ! isapprox(f64(A),A64,rtol=100*max(eps(T),eps(Float64)))
            error("error: ($m,$n)")
        end
    end
    @info "getrf:  tgnrc/tblas=$(tgnrc/tblas)"
    true
end



#
# Test gtrsm! for a couple of random data,
# compare timing between generic and blas  implementations
#
function ttrsm(T=Float64)
    tblas=0.0
    tgnrc=0.0
    ldx()=rand(0:5)
    for m in rand(1:50,15)
        for n in rand(1:50,15)
            for side in ['r','l']
                for uplo  in ['l','u']
                    for transa in ['t','n']
                        for diag  in ['u','n']

                            k= side=='l' ? m : n

                            lda=ldx()+k
                            A=-rand(T,lda*k)
                            rA=strided_reshape(A,lda,k,k)
                            for i=1:k
                                rA[i,i]=10.0 #Diagonal(10I,size(A[1,1],1))
                            end

                        
                            
                            ldb= ldx()+m
                            B=rand(T,ldb*n)
                            
                            alpha=rand(T)
                            alpha64=f64(alpha)
                            B64=f64(B)
                            A64=f64(A)

                            tblas += @elapsed dtrsm!(side, uplo, transa, diag, m,n, alpha64, A64, lda, B64,ldb)
                            tgnrc += @elapsed gtrsm!(side, uplo, transa, diag, m,n, alpha, A, lda, B,ldb)
                            if ! isapprox(f64(B),B64,rtol=10*max(eps(T),eps(Float64)))
                                error(" error for side $side, uplo $uplo, transa $transa diag $diag $n, $m")
                            end
                        end
                    end
                end
            end
        end
    end
    @info "trsm:  tgnrc/tblas=$(tgnrc/tblas)"
    true
end
    
#
# Test glaswp! for a couple of random data,
# compare timing between generic and blas  implementations
#
function tlaswp(T=Float64)
    tblas=0.0
    tgnrc=0.0
    for n in rand(1:50,15)
        ipiv=shuffle(1:n)
        A=rand(T,n)
        A64=f64(A)
        tblas += @elapsed dlaswp!(A64,n,1,n,ipiv)
        tgnrc += @elapsed glaswp!(A,n,1,n,ipiv)
        if ! (f64(A)≈ A64)
            error("laswp failed for $n")
        end
    end
    @info "laswp:  tgnrc/tblas=$(tgnrc/tblas)"
    true
end

#
# Run all tests for type T
#
function test_all_T(T=Float64)
    @info "$T:"
    @testset "test $(T)" begin
        @test tgemm(T)
        @test tgemv(T)
        @test tgetrf(T)
        @test ttrsm(T)
        @test tlaswp(T)
    end
end

#
# Run tests for selected types
#
function test_all()
    @testset "setup" begin
        tf64()
    end
    test_all_T(Float64)
    test_all_T(Float32)
    test_all_T(ComplexF64)
    test_all_T(ComplexF32)
    test_all_T(ForwardDiff.Dual{Float64,Float64,1})
    test_all_T(MultiFloats.Float64x2)
end

_test()=test_all()

_test()
end

