module SparseCSCInterface
using SparseArrays, LinearAlgebra
using ..SpkGraph: Graph
using ..SpkOrdering: Ordering
using ..SpkETree: ETree
using ..SpkSparseBase: _SparseBase
using ..SpkSparseSolver: SparseSolver, findorder!,symbolicfactor!,inmatrix!,factor!,triangularsolve!
import ..SpkSparseSolver: solve!
import ..SpkSparseBase: _inmatrix!


function Graph(m::SparseMatrixCSC{FT,IT}, diagonal=false) where {FT,IT}
    nv = size(m,1)
    nrows = size(m,2)
    ncols = size(m,1)
    colptr = SparseArrays.getcolptr(m)
    rowval = SparseArrays.getrowval(m)


    if (diagonal)
        nedges = nnz(m)
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
        nedges = nnz(m) - dedges
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




function _SparseBase(m::SparseMatrixCSC{FT,IT}) where {IT,FT}
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


function _inmatrix!(s::_SparseBase{IT, FT}, m::SparseMatrixCSC{FT,IT}) where {IT, FT}
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



function SparseSolver(m::SparseMatrixCSC)
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


function solve!(s::SparseSolver{IT}, rhs) where {IT}
    findorder!(s) || ErrorException("Finding Order.")
    symbolicfactor!(s) || ErrorException("Symbolic Factorization.")
    inmatrix!(s) || ErrorException("Matrix input.")
    factor!(s) || ErrorException("Numerical Factorization.")
    temp=copy(rhs)
    triangularsolve!(s,temp) || ErrorException("Triangular Solve.")
    return temp
end

#########################################################################
# SparspakLU

"""
    sparspaklu(m; factorize=true)

If `factorize==true`, calculate LU factorization using Sparspak. Steps
are `findorder`, `symbolicfactor`, `factor`.

If   `factorize==false`,   ordering,    symbolic   factorization   and
factorization are delayed to a subsequent call to `sparspaklu!`.

Returns  a  `SparseSolver` instance in the respective state, 
which has methods for `LinearAlgebra.ldiv!` and "backslash".
"""
function sparspaklu(m::SparseMatrixCSC;factorize=true)
    lu=SparseSolver(m)
    if factorize
        findorder!(lu) || ErrorException("Finding Order.")
        symbolicfactor!(lu) || ErrorException("Symbolic Factorization.")
        inmatrix!(lu) || ErrorException("Matrix input.")
        factor!(lu) || ErrorException("Numerical Factorization.")
    end
    lu
end


"""
    sparspaklu!(lu,m)

Calculate   LU   factorization,    reusing   ordering   and   symbolic
factorization from lu, if that was previously calculated.

Currently, it is  assumed that, if size and number  of nonzeros didn't
change, the  sparsity patterns of `m`  and `p` are the  same, probably
leading to errors elsewhere if the patterns nevertheless differ.

"""
function sparspaklu!(lu::SparseSolver, m::SparseMatrixCSC)
    # jf: Do we need a better test here ? Not sure as that may be expensive.
    if lu.slvr.n != size(m,1) ||   lu.slvr.n != size(m,2) ||     lu.slvr.nnz != nnz(m)
        lu=SparseSolver(m)
    end        
    lu.p=m
    lu._orderdone    || ( findorder!(lu)      || ErrorException("Finding Order.") )
    lu._symbolicdone || ( symbolicfactor!(lu) || ErrorException("Symbolic Factorization.") )
    lu._inmatrixdone = false
    lu._factordone = false
    lu._trisolvedone = false
    inmatrix!(lu) || ErrorException("Matrix input.")
    factor!(lu) || ErrorException("Numerical Factorization.")
    lu
end


"""
    ldiv(u,lu::SparseSolver,v)

Left division for SparseSolver
"""
function LinearAlgebra.ldiv!(u, lu::SparseSolver, v)
    u.=v
    triangularsolve!(lu,u) || ErrorException("Triangular Solve.")
    lu._trisolvedone = false
    u
end

"""
    ldiv(lu::SparseSolver,v)

Overwriting left division for SparseSolver.
"""
function LinearAlgebra.ldiv!(lu::SparseSolver, v)
    triangularsolve!(lu,v) || ErrorException("Triangular Solve.")
    lu._trisolvedone = false
    v
end

"""
    \\(lu::SparseSolver,v)

"Backslash" operator for sparse solver
"""
Base.:\(lu::SparseSolver, v)=ldiv!(lu,copy(v))

end
