module SparseCSCInterface
import SparseArrays, LinearAlgebra
using ..SpkGraph: Graph
using ..SpkOrdering: Ordering
using ..SpkETree: ETree
using ..SpkSparseBase: _SparseBase
using ..SpkSparseSolver: SparseSolver, findorder!,symbolicfactor!,inmatrix!,factor!,triangularsolve!
import ..SpkSparseSolver: solve!
import ..SpkSparseBase: _inmatrix!


function Graph(m::SparseArrays.SparseMatrixCSC{FT,IT}, diagonal=false) where {FT,IT}
    nv = size(m,1)
    nrows = size(m,2)
    ncols = size(m,1)
    colptr = SparseArrays.getcolptr(m)
    rowval = SparseArrays.getrowval(m)


    if (diagonal)
        nedges = SparseArrays.nnz(m)
    else
        dedges=0
        for i in 1:ncols
            for iptr in colptr[i]:colptr[i+1]-1
                if  rowval[iptr]==i
                    dedges+=1
                    continue
                end
            end
        end
        nedges = SparseArrays.nnz(m) - dedges
    end
    
    #jf if diagonal == true, we possibly can just use colptr & rowval
    #jf and skip the loop
   
    xadj = fill(zero(IT), nv + 1)
    adj = fill(zero(IT), nedges)

    k = 1
    for i in 1:ncols
        xadj[i] = k
        for iptr in colptr[i]:colptr[i+1]-1
            j = rowval[iptr]
            if (i != j || diagonal)
                adj[k] = j
                k = k + 1
            end
        end
    end
    
    xadj[ncols+1] = k
    
    return Graph(nv, nedges, nrows, ncols, xadj, adj)
end


function _SparseBase(m::SparseArrays.SparseMatrixCSC{FT,IT}) where {IT,FT}
    maxblocksize = 30   # This can be set by the user
    
    tempsizeneed = zero(IT)
    n = size(m,2)
    nnz = SparseArrays.nnz(m)
    nnzl = zero(IT)
    nsub = zero(IT)

    nsuper = zero(IT)
    if (n > zero(IT))
        nsuper = 1
    end

    factorops = zero(FT)
    solveops = zero(FT)
    realstore = zero(FT)
    integerstore = zero(FT)
    errflag = 0

    order = Ordering(n)   # ordering object for the solver
    g = Graph(m)
    t = ETree(n)

    colcnt = IT[]
    snode = IT[]
    xsuper = IT[]
    xlindx = IT[]
    lindx = IT[]
    xlnz = IT[]
    xunz = IT[]
    ipiv = IT[]
    lnz = FT[]
    unz = FT[]

    return _SparseBase(order, t, g, errflag, n, nnz, nnzl, nsub, nsuper, maxblocksize,
        tempsizeneed, factorops, solveops, realstore, integerstore,
        colcnt, snode, xsuper, xlindx, lindx, xlnz, xunz, ipiv,
        lnz, unz)
end


function _inmatrix!(s::_SparseBase{IT, FT}, m::SparseArrays.SparseMatrixCSC{FT,IT}) where {IT, FT}
    if (s.n == 0)
        error("An empty problem. No matrix.")
        return false
    end

    s.lnz .= zero(FT)
    s.unz .= zero(FT)
    s.ipiv .= zero(IT)

    function doit(ncols, colptr, rowval, cinvp, rinvp, snode, xsuper, xlindx, lindx, nzval, xlnz, lnz, xunz, unz)
        for i in 1:ncols
            for iptr in colptr[i]:colptr[i+1]-1
                inew = rinvp[rowval[iptr]];
                jnew = cinvp[i]
                value = nzval[iptr]
## jf: all of this could go into  a function, so we can keep things synced                
                if (inew >= xsuper[snode[jnew]])
#               Lies in L.  get pointers and lengths needed to search
#               column jnew of L for location l(inew, jnew).
                    jsup = snode[jnew];
                    fstcol = xsuper[jsup]
                    fstsub = xlindx[jsup]
                    lstsub = xlindx[jsup + 1] - 1
                    nnzloc = 0;
                    for nxtsub in fstsub:lstsub
                        irow = lindx[nxtsub]
                        if  (irow > inew)
                            error("No space for matrix element ($(inew), $(jnew)).")
                            return false
                        end
                        if  (irow == inew)
#                       find a proper offset into lnz and increment by value
                            _p = xlnz[jnew] + nnzloc
                            lnz[_p] += value
                            break
                        end
                        nnzloc = nnzloc + 1
                    end
                else
#               Lies in U
                    jsup = snode[inew]
                    fstcol = xsuper[jsup]
                    lstcol = xsuper[jsup + 1] - 1
                    width = lstcol - fstcol + 1
                    lstsub = xlindx[jsup + 1] - 1
                    fstsub = xlindx[jsup] + width
                    nnzloc = 0;
                    for nxtsub in fstsub:lstsub
                        irow = lindx[nxtsub]
                        if  (irow > jnew)
                            error("No space for matrix element ($(inew), $(jnew)).")
                            return false
                        end
                        if  (irow == jnew)
#                       find a proper offset into unz and increment by value
                            _p = xunz[inew] + nnzloc
                            unz[_p] += value
                            break
                        end
                        nnzloc = nnzloc + 1
                    end
                end
            end
        end
        return true
    end
    return doit(size(m,1), m.colptr, m.rowval, s.order.rinvp, s.order.cinvp, s.snode, s.xsuper, s.xlindx, s.lindx, m.nzval, s.xlnz, s.lnz, s.xunz, s.unz)
end

function SparseSolver(m::SparseArrays.SparseMatrixCSC{FT,IT}) where {FT,IT}
    ma = size(m,2)
    na = size(m,1)
    mc = 0
    nc = 0
    n = ma
    slvr = _SparseBase(m)
    _orderdone = false
    _symbolicdone = false
    _inmatrixdone = false
    _factordone = false
    _trisolvedone = false
    _refinedone = false
    _condestdone = false
    return SparseSolver(m, slvr, n, ma, na, mc, nc, _inmatrixdone, _orderdone, _symbolicdone, _factordone, _trisolvedone, _refinedone, _condestdone)
end


"""
    solve!(s,rhs)

Solves linear system defined with sparse solver and provides the solution in rhs.
"""
function solve!(s::SparseSolver{IT,FT}, rhs) where {IT,FT}
    findorder!(s) || error("Finding Order.")
    symbolicfactor!(s) || error("Symbolic Factorization.")
    inmatrix!(s) || error("Matrix input.")
    factor!(s) || error("Numerical Factorization.")
    triangularsolve!(s,rhs) || error("Triangular Solve.")
    return true
end

#########################################################################
# SparspakLU

"""
    sparspaklu(m::SparseMatrixCSC; factorize=true) -> lu::SparseSolver

If `factorize==true`, calculate LU factorization using Sparspak. Steps
are `findorder`, `symbolicfactor`, `factor`.

If   `factorize==false`,   ordering,    symbolic   factorization   and
factorization are delayed to a subsequent call to `sparspaklu!`.

Returns  a  `SparseSolver` instance in the respective state, 
which has methods for `LinearAlgebra.ldiv!` and "backslash".
"""
function sparspaklu(m::SparseArrays.SparseMatrixCSC{FT,IT};factorize=true) where {FT,IT}
    lu=SparseSolver(m)
    if factorize
        findorder!(lu) || error("Finding Order.")
        symbolicfactor!(lu) || error("Symbolic Factorization.")
        inmatrix!(lu) || error("Matrix input.")
        factor!(lu) || error("Numerical Factorization.")
    end
    lu
end


"""
    sparspaklu!(lu::SparseSolver, m::SparseMatrixCSC; allow_pattern_change=true) -> lu::SparseSolver

Calculate LU factorization of a sparse matrix `m`, reusing ordering and symbolic
factorization `lu`, if that was previously calculated.

If `allow_pattern_change = true` (the default) the sparse matrix `m` may have a nonzero pattern
different to that of the matrix used to create the LU factorization `lu`, in which case the ordering
and symbolic factorization are updated.

If `allow_pattern_change = false` an error is thrown if the nonzero pattern of `m` is different to that 
of the matrix used to create the LU factorization `lu`.

If `lu` has not been factorized (ie it has just been created with option `factorize = false`) then 
`lu` is always updated from `m` and `allow_pattern_change` is ignored.

"""
function sparspaklu!(lu::SparseSolver{IT,FT}, m::SparseArrays.SparseMatrixCSC{FT,IT}; allow_pattern_change=true) where {FT,IT}
   
    pattern_changed = (SparseArrays.getcolptr(m) != SparseArrays.getcolptr(lu.p)) || (SparseArrays.getrowval(m) != SparseArrays.getrowval(lu.p))

    if pattern_changed
        if allow_pattern_change || !lu._symbolicdone
            copy!(lu, SparseSolver(m))
        else
            error("'allow_pattern_change=false', but sparsity pattern of matrix 'm' does not match that used to create 'lu'")
        end
    end
   
    lu.p=m
    lu._orderdone    || ( findorder!(lu)      || error("Finding Order.") )
    lu._symbolicdone || ( symbolicfactor!(lu) || error("Symbolic Factorization.") )
    lu._inmatrixdone = false
    lu._factordone = false
    lu._trisolvedone = false
    inmatrix!(lu) || error("Matrix input.")
    factor!(lu) || error("Numerical Factorization.")
    lu
end

"""
    ldiv!(u, lu::SparseSolver{IT,FT}, v) where {IT,FT}

Left division for SparseSolver.

The solution is returned in `u`. The right hand side vector is `v`.
"""
function LinearAlgebra.ldiv!(u, lu::SparseSolver{IT,FT}, v) where {IT,FT}
    u.=v
    return ldiv!(lu, u)
end

"""
    ldiv!(lu::SparseSolver{IT,FT}, v) where {IT,FT}

Overwriting left division for SparseSolver.

The solution is returned in `v`, which is also the right hand side vector.
"""
function LinearAlgebra.ldiv!(lu::SparseSolver{IT,FT}, v) where {IT,FT}
    solve!(lu, v) || error("Triangular Solve.")
    lu._trisolvedone = false
    v
end

"""
    \\(lu::SparseSolver,v)

"Backslash" operator for sparse solver
"""
Base.:\(lu::SparseSolver{IT,FT}, v) where {IT,FT}=LinearAlgebra.ldiv!(lu,copy(v))

end
