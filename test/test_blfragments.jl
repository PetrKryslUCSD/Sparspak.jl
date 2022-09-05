module test_blfragments
using Test
using LinearAlgebra
using Random
using LinearAlgebra.BLAS: BlasInt
using Sparspak.SpkSpdMMops:  dgemm!,dgemv!,dgetrf!,dtrsm!,dlaswp!
using Sparspak.GenericBlasLapackFragments:  ggemm!,ggemv!,ggetrf!,gtrsm!,glaswp!


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
    for m in rand(1:50,15)
        for n in rand(1:50,15)
            for k in rand(1:50,15)
                
                A=rand(T,m,k)
                B=rand(T,k,n)
                C0=rand(T,m,n)
                α=rand(T)
                β=rand(T)
                

                vA=vec(A)
                A64=f64(A)
                vA64=vec(A64)

                vB=vec(B)
                B64=f64(B)
                vB64=vec(B64)

                α64=f64(α)
                β64=f64(β)
                
                C=copy(C0)
                vC=vec(C)
                C64=f64(C)
                vC64=vec(C64)
                tblas+=@elapsed dgemm!('n', 'n', m,n,k, α64 ,vA64, m ,vB64,k, β64,vC64,m)
                tgnrc+=@elapsed ggemm!('n', 'n', m,n,k, α ,vA, m ,vB,k, β,vC,m)
                if ! isapprox(f64(C),C64,rtol=10*max(eps(T),eps(Float64)))
                    error("error in n-n ($n, $m, $k)")
                end
                
                C=copy(C0)
                vC=vec(C)
                C64=f64(C)
                vC64=vec(C64)
                tblas+=@elapsed dgemm!('n', 't', m,n,k, α64 ,vA64, m ,vB64, n, β64,vC64,m)
                tgnrc+=@elapsed ggemm!('n', 't', m,n,k, α ,vA, m ,vB,n, β,vC,m)
                if ! isapprox(f64(C),C64,rtol=10*max(eps(T),eps(Float64)))
                    error("error in n-t ($n, $m, $k)")
                end
                
                C=copy(C0)
                vC=vec(C)
                C64=f64(C)
                vC64=vec(C64)
                tblas+=@elapsed dgemm!('t', 'n', m,n,k, α64 ,vA64, k ,vB64, k, β64,vC64,m)
                tgnrc+=@elapsed ggemm!('t', 'n', m,n,k, α ,vA, k ,vB,k, β,vC,m)
                if ! isapprox(f64(C),C64,rtol=10*max(eps(T),eps(Float64)))
                    error("error in t-n ($n, $m, $k)")
                end
                
                C=copy(C0)
                vC=vec(C)
                C64=f64(C)
                vC64=vec(C64)
                tblas+=@elapsed dgemm!('t', 't', m,n,k, α64 ,vA64, k ,vB64, n, β64,vC64,m)
                tgnrc+=@elapsed ggemm!('t', 't', m,n,k, α ,vA, k ,vB,n, β,vC,m)
                if ! isapprox(f64(C),C64,rtol=10*max(eps(T),eps(Float64)))
                    error("error in t-t ($n, $m, $k)")
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
    for m in rand(1:50,15)
        for n in rand(1:50,15)
            A=rand(T,m,n)
            α=rand(T)
            β=rand(T)
            
            vA=vec(A)
            A64=f64(A)
            vA64=vec(A64)

            α64=f64(α)
            β64=f64(β)


            X=rand(T,n)
            Y=rand(T,m)
            X64=f64(X)
            Y64=f64(Y)
            tblas+=@elapsed dgemv!('n', m,n, α64 ,vA64, m ,X64, β64,Y64)
            tgnrc+=@elapsed ggemv!('n', m,n, α ,vA, m ,X, β,Y)
            if ! isapprox(f64(Y),Y64,rtol=10*max(eps(T),eps(Float64)))
                error("error in n ($n, $m)")
            end

            X=rand(T,m)
            Y=rand(T,n)
            X64=f64(X)
            Y64=f64(Y)
            tblas+=@elapsed dgemv!('t', m,n, α64 ,vA64, m ,X64, β64,Y64)
            tgnrc+=@elapsed ggemv!('t', m,n, α ,vA, m ,X, β,Y)
            if ! isapprox(f64(Y),Y64,rtol=10*max(eps(T),eps(Float64)))
                error("error in t ($n, $m)")
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
    for n in rand(1:50,15)
        m=n
        A=-rand(T,m,n)
        for i=1:n
            A[i,i]=2.0*max(m,n)  
        end
        
        ipiv=zeros(BlasInt,min(m,n))
        ipiv64=zeros(BlasInt,min(m,n))

        vA=vec(A)
        A64=f64(A)
        vA64=vec(A64)

        tblas+=@elapsed Alu64=dgetrf!(m,n,vA64,n,ipiv64)
        tgnrc+=@elapsed Alu=ggetrf!(m,n,vA,n,ipiv)
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
    for m in rand(1:50,15)
        for n in rand(1:50,15)
            for side in ['r','l']
                for transa in ['n']
                    for diag  in ['n','u']
                        
                        k= side=='l' ? m : n
                        A=rand(T,k,k)
                        for i=1:k
                            A[i,i]=10.0 #Diagonal(10I,size(A[1,1],1))
                        end
                        B=rand(T,m,n)
                        alpha=rand(T)
                        alpha64=f64(alpha)
                        vA=vec(A)
                        vA64=vec(f64(A))
                        gB=copy(B)
                        vgB=vec(gB)
                        B64=f64(B)
                        vB64=vec(B64)
                        
                        tblas += @elapsed dtrsm!(side, 'l', transa, 'n', m,n, alpha64, vA64, k, vB64,m)
                        tgnrc += @elapsed gtrsm!(side, 'l', transa, 'n', m,n, alpha, vA, k, vgB,m)
                        if ! isapprox(f64(gB),B64,rtol=10*max(eps(T),eps(Float64)))
                            error(" error for uplo l, side $side, transa $transa diag $diag $n, $m")
                        end
                        
                        vA=vec(A)
                        vA64=vec(f64(A))
                        gB=copy(B)
                        vB=vec(B)
                        vgB=vec(gB)
                        B64=f64(B)
                        vB64=vec(B64)
                        
                        
                        tblas += @elapsed dtrsm!(side, 'u', transa, diag, m,n, alpha64, vA64, k, vB64,m)
                        tgnrc += @elapsed gtrsm!(side, 'u', transa, diag, m,n, alpha, vA, k, vgB,m)
                        if ! isapprox(f64(gB),B64,rtol=10*max(eps(T),eps(Float64)))
                            error(" error for uplo u, side $side, transa $transa diag $diag $n, $m")
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

