#
# Test the generic blas/lapack  code against blas calls in Sparspak
#

module test_blfragments
using Test
using LinearAlgebra
using Random
using LinearAlgebra.BLAS: BlasInt

# These will be called for Float64, calling back to BLAS
using Sparspak.SpkSpdMMops:  _gemm!,_gemv!,_getrf!,_trsm!,_laswp!

# These are the Julia based replacements which are explicitely called.
using Sparspak.GenericBlasLapackFragments:  ggemm!,ggemv!,ggetrf!,gtrsm!,glaswp!

#
# Two potentialy useful extensions for numeber types to be tested
#
using ForwardDiff
using MultiFloats


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

r(x)=round(x,digits=3)

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

function buflayout(m,n,mode)
    if mode==:contiguous
        lda=m
        la=m*n
    elseif mode==:strided
        lda=m+rand(1:5)
        la=lda*n
    elseif mode==:strided_chopped
        lda=m+rand(1:5)
        la=lda*n - rand(1:lda-m)
    elseif mode==:mixed
        lda=m+rand(0:5)
        la=lda*n - rand(0:lda-m)
    else
        error("Wrong buflayout mode: $(mode)")
    end
    la,lda
end


#
# Test ggemm! for a couple of random data,
# compare timing between generic and blas  implementations.
# 
#
function _tgemm(T=Float64;N=15,mode=:random)
    tblas=0.0
    tgnrc=0.0
    for m in rand(1:50,N)
        for n in rand(1:50,N)
            for k in rand(1:50,N)
                for transA in ['n','t']
                    for transB in ['n','t']

                        if transA=='n'
                            la,lda=buflayout(m,k,mode)
                        else
                            la,lda=buflayout(k,m,mode)
                        end
                        A=rand(T,la)
                        
                        if transB=='n'
                            lb,ldb=buflayout(k,n,mode)
                        else
                            lb,ldb=buflayout(n,k,mode)
                        end
                        B=rand(T,lb)

                        lc,ldc=buflayout(m,n,mode)
                        C=rand(T,lc)


                        α=-one(T)
                        β=one(T)
                        
                        
                        A64=f64(A)
                        C64=f64(C)
                        B64=f64(B)

                        α64=f64(α)
                        β64=f64(β)
                        
                        
                        
                        tblas+=@elapsed _gemm!(transA,transB, m,n,k, α64 ,A64, lda ,B64,ldb, β64,C64,ldc)
                        tgnrc+=@elapsed ggemm!(transA,transB, m,n,k, α ,A, lda ,B,ldb, β,C,ldc)
                        if ! isapprox(f64(C),C64,rtol=100*max(eps(T),eps(Float64)))
                            error("error in $transA-$transB ($n, $m, $k)")
                        end
                    end
                end
            end
        end
    end
    tgnrc,tblas
end

const N_compile=10
function tgemm(T=Float64)
    _tgemm(T,N=N_compile,mode=:mixed)
    ctgnrc,ctblas=_tgemm(T,N=15,mode=:contiguous)
    stgnrc,stblas=_tgemm(T,N=15,mode=:strided)
    sctgnrc,sctblas=_tgemm(T,N=15,mode=:strided_chopped)
    @info "gemm:  tgnrc/tblas: contig: $(ctgnrc/ctblas |>r) strided: $(stgnrc/stblas |>r) StridedReshape: $(sctgnrc/sctblas|>r)"
    true
end

#
# Test ggemv! for a couple of random data,
# compare timing between generic and blas  implementations
#
function _tgemv(T=Float64;N=15,mode=:mixed)
    tblas=0.0
    tgnrc=0.0
    for m in rand(1:50,N)
        for n in rand(1:50,N)
            for transA in ['n','t']
                
                if transA=='n'
                    la,lda=buflayout(m,n,mode)
                else
                    la,lda=buflayout(n,m,mode)
                end
                A=rand(T,la)
   
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
                tblas+=@elapsed _gemv!(transA, m,n, α64 ,A64, m ,X64, β64,Y64)
                tgnrc+=@elapsed ggemv!(transA, m,n, α ,A, m ,X, β,Y)
                if ! isapprox(f64(Y),Y64,rtol=100*max(eps(T),eps(Float64)))
                    error("error in $transA ($n, $m)")
                end
            end
        end
    end
    tgnrc,tblas
end

function tgemv(T=Float64)
    _tgemv(T,N=N_compile,mode=:mixed)
    ctgnrc,ctblas=_tgemv(T,N=30,mode=:contiguous)
    stgnrc,stblas=_tgemv(T,N=30,mode=:strided)
    sctgnrc,sctblas=_tgemv(T,N=30,mode=:strided_chopped)
    @info "gemv:  tgnrc/tblas: contig: $(ctgnrc/ctblas |>r) strided: $(stgnrc/stblas |>r) StridedReshape: $(sctgnrc/sctblas|>r)"
    true
end



#
# Test ggetrf! for a couple of random data,
# compare timing between generic and blas  implementations
#
function _tgetrf(T=Float64;N=25, mode=:mixed)
    tblas=0.0
    tgnrc=0.0
    for n in rand(1:100,N)
        m=n
        la,lda=buflayout(m,n,mode)
        A=-rand(T,la)
        rA=strided_reshape(A,lda,m,m)
        for i=1:n
            rA[i,i]=2.0*max(m,n)  
        end
        
        ipiv=zeros(BlasInt,min(m,n))
        ipiv64=zeros(BlasInt,min(m,n))
        
        A64=f64(A)

        tblas+=@elapsed Alu64=_getrf!(m,n,A64,lda,ipiv64)
        tgnrc+=@elapsed Alu=ggetrf!(m,n,A,lda,ipiv)
        if ! isapprox(f64(A),A64,rtol=100*max(eps(T),eps(Float64)))
            error("error: ($m,$n)")
        end
    end
    tgnrc,tblas
end

function tgetrf(T=Float64)
    _tgetrf(T,N=N_compile,mode=:mixed)
    ctgnrc,ctblas=_tgetrf(T,N=15,mode=:contiguous)
    stgnrc,stblas=_tgetrf(T,N=15,mode=:strided)
    sctgnrc,sctblas=_tgetrf(T,N=15,mode=:strided_chopped)
    @info "getrf: tgnrc/tblas: contig: $(ctgnrc/ctblas |>r) strided: $(stgnrc/stblas |>r) StridedReshape: $(sctgnrc/sctblas|>r)"
    true
end

#
# Test gtrsm! for a couple of random data,
# compare timing between generic and blas  implementations
#
function _ttrsm(T=Float64;N=15,mode=:mixed)
    tblas=0.0
    tgnrc=0.0
    ldx()=rand(0:5)
    for m in rand(1:50,N)
        for n in rand(1:50,N)
            for side in ['r','l']
                for uplo  in ['l','u']
                    for transa in ['t','n']
                        for diag  in ['u','n']

                            k= side=='l' ? m : n
#                            lda=ldx()+k
#                            A=-rand(T,lda*k)

                            la,lda=buflayout(k,k,mode)
                            A=-rand(T,la)
                            rA=strided_reshape(A,lda,k,k)
                            for i=1:k
                                rA[i,i]=10.0 #Diagonal(10I,size(A[1,1],1))
                            end

#                            ldb= ldx()+m
#                            B=rand(T,ldb*n)

                        
                            lb,ldb=buflayout(m,n,mode)
                            B=rand(T,lb)
                            
                            alpha=rand(T)
                            alpha64=f64(alpha)
                            B64=f64(B)
                            A64=f64(A)

                            tblas += @elapsed _trsm!(side, uplo, transa, diag, m,n, alpha64, A64, lda, B64,ldb)
                            tgnrc += @elapsed gtrsm!(side, uplo, transa, diag, m,n, alpha, A, lda, B,ldb)
                            if ! isapprox(f64(B),B64,rtol=100*max(eps(T),eps(Float64)))
                                error(" error for side $side, uplo $uplo, transa $transa diag $diag $n, $m")
                            end
                        end
                    end
                end
            end
        end
    end
    tgnrc,tblas
end

function ttrsm(T=Float64)
    _ttrsm(T,N=N_compile,mode=:mixed)
    ctgnrc,ctblas=_ttrsm(T,N=15,mode=:contiguous)
    stgnrc,stblas=_ttrsm(T,N=15,mode=:strided)
    sctgnrc,sctblas=_ttrsm(T,N=15,mode=:strided_chopped)
    @info "trsm:  tgnrc/tblas: contig: $(ctgnrc/ctblas |>r) strided: $(stgnrc/stblas |>r) StridedReshape: $(sctgnrc/sctblas|>r)"
    true
end

#
# Test glaswp! for a couple of random data,
# compare timing between generic and blas  implementations
#
function _tlaswp(T=Float64;N=15)
    tblas=0.0
    tgnrc=0.0
    for n in rand(1:50,N)
        ipiv=shuffle(1:n)
        A=rand(T,n)
        A64=f64(A)
        tblas += @elapsed _laswp!(A64,n,1,n,ipiv)
        tgnrc += @elapsed glaswp!(A,n,1,n,ipiv)
        if ! (f64(A)≈ A64)
            error("laswp failed for $n")
        end
    end
    tgnrc,tblas
end


function tlaswp(T=Float64)
    _tlaswp(T,N=N_compile)
    tgnrc,tblas=_tlaswp(T,N=15)
    @info "laswp:  tgnrc/tblas=$(tgnrc/tblas |>r)"
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
        @test tf64()
    end
    test_all_T(Float64)
    test_all_T(Float32)
    test_all_T(ComplexF64)
    test_all_T(ComplexF32)
    test_all_T(ForwardDiff.Dual{Float64,Float64,1})
    test_all_T(ForwardDiff.Dual{Float64,Float64,2})
    test_all_T(MultiFloats.Float64x1)
    test_all_T(MultiFloats.Float64x2)
end

_test()=test_all()

_test()

end

