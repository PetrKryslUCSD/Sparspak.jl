module SpkLUFactor

using ..SpkSpdMMops: _mmpyi!, _assmb!, _luswap!, _ldindx!, _igathr!, _gemm!, _trsm!, _getrf!, _laswp!, _gemv!


#      this subroutine computes an LU factorization of a sparse
#      matrix with symmetric structure.  the computation is
#      organized around kernels that perform supernode - to - supernode
#      updates.  level - 3 blas routines used whenever possible.
#      each supernode is a dense trapezoidal submatrix stored in
#      a rectangular array.
# #
#  input parameters:
#      nsuper - number of supernode.
#      xsuper - supernode partition.
#      snode - maps each column to the supernode containing it.
#      (xlindx, lindx) - row indices for each supernode (including
#                          the diagonal elements).
#      (xlnz, lnz) - on input, contains the lower half of the matrix
#                          to be factored and the diagonal block.
#      (xunz, unz) - on input, contains the upper half of the matrix
#                          to be factored.
# #
#  output parameters:
#      lnz - on output, contains the lower half of the
#                          cholesky factor.
#      unz - on output, contains the upper half of the
#                          cholesky factor.
#      iflag - error flag.
#                              0: successful factorization.
# - 1: nonpositive diagonal encountered,
#                                 matrix is not positive definite.
# - 2: insufficient working storage
#                                 [temp(*)].
#      ipvt - row pivoting information.
# #
#  local variables:
#      link - a linked list that stores the supernode
#                          that are ready to modify jsup.
#                          if link(jsup) = k,  supernodes k,
#                          link(k), link(link(k)), ... all modify
#                          supernode jsup.
#      length - length of the active portion of each supernode.
#                          ie. the length that hasn"t effected a diagonal
#                          block yet.
#      map - vector of size n into which the global
#                          indices are scattered.  This array holds the
#                          distance of row k from the bottom of the
#                          supernode if it were compressed.
#      relind - maps locations in the updating columns to
#                          the corresponding locations in the updated
#                          columns.  (relind is gathered from map).
#                          more specifically, this is an array of length
#                          lenght(ksup), and only stores the elements
#                          from map coresponding to union of row indexes
#                          of jsup and ksup.
#      temp - real vector for accumulating updates.  must
#                           accomodate all columns of a supernode.

function _lufactor!(n::IT, nsuper::IT, xsuper::Vector{IT}, snode::Vector{IT}, xlindx::Vector{IT}, lindx::Vector{IT}, xlnz::Vector{IT}, lnz::Vector{FT}, xunz::Vector{IT}, unz::Vector{FT}, ipvt::Vector{IT}) where {IT, FT}
    @assert length(xsuper) == (nsuper + 1)
    @assert length(snode) == n
    @assert length(xlindx) == (nsuper + 1)
    @assert length(xlnz) == (n + 1)
    @assert length(xunz) == (n + 1)
    @assert length(ipvt) == n

    ONE = one(FT)
    
    link = zeros(IT, nsuper)
    lngth = zeros(IT, nsuper)
#
    iflag = zero(IT);   tmpsiz = zero(IT)
#   Initialization
    for i in 1:nsuper
        link[i] = 0
        lngth[i] = xlindx[i + 1] - xlindx[i]
        width = xsuper[i + 1] - xsuper[i]
        need = width * (xlnz[xsuper[i] + 1] - xlnz[xsuper[i]])
        if (need > tmpsiz) tmpsiz = need; end
    end

    map = zeros(IT, n)
    relind = zeros(IT, n)
    temp = zeros(FT, tmpsiz)
    vtemp = view(temp, 1:length(temp))

#   For each supernode jsup ...
    for jj in 1:nsuper
        jsup = jj
        # fj     ...  first row / column of jsup.
        # lj     ...  last row / column of jsup.
        # nj     ...  number of rows / columns in jsup.
        # jlen   ...  lngth of jsup.
        # jxpnt  ...  pointer to index of first nonzero in
        #             jsup.
        # jlpnt  ...  pointer to location of first nonzero
        #             in column fj.
        # jupnt  ...  pointer to location of first nonzero
        #             in row fj.
        fj    = xsuper[jsup]
        lj    = xsuper[jsup + 1] - 1
        nj    = lj - fj + 1
        jlen  = xlnz[fj + 1] - xlnz[fj]
        jxpnt = xlindx[jsup]
        jlpnt = xlnz[fj]
        jupnt = xunz[fj]
#       Set up map(*) to map the entries in update columns to their
#       corresponding positions in updated columns, relative to the bottom of
#       each updated column.
        _ldindx!(jlen, view(lindx, jxpnt:length(lindx)), map)
#       For every supernode ksup in row(jsup) ...
        while (true)
#           Wait for something to appear in the list, or stop if there can"t be
#           anything more

#           Take out the first item in the list.  MUST BE DONE atomically
            ksup = link[jsup]
            if (ksup != 0)
#               Remove the head of the list.
                link[jsup] = link[ksup]
                link[ksup] = 0
            end
            if (ksup == 0) break; end
#             get info about the cmod(jsup, ksup) update.
#
#             fk     ...  first row / column of ksup.
#             nk     ...  number of rows / columns in ksup.
#             ksuplen  ...  lngth of ksup.
#             klen   ...  lngth of active portion of ksup.
#             kxpnt  ...  pointer to index of first nonzero in
#                         active portion of ksup.
#             klpnt  ...  pointer to location of first nonzero in
#                         active portion of column fk.
#             kupnt  ...  pointer to location of first nonzero in
#                         active portion of row fk.
            fk  = xsuper[ksup]
            lk  = xsuper[ksup + 1] - 1
            nk = lk - fk + 1
            ksuplen  = xlnz[fk + 1] - xlnz[fk]
            klen   = lngth[ksup]
            kxpnt  = xlindx[ksup + 1] - klen
            klpnt = xlnz[fk + 1] - klen
            kupnt = xunz[fk + 1] - klen
#           Perform cmod(jsup, ksup), with special cases handled differently.
#
#           Each cmod is broken up into two stages: first perform cmod on the
#           columns, perform cmod on the rows.
#           nups  ...  number of rows / columns updated.
            if  (klen == jlen)
#               Dense cmod(jsup, ksup). jsup and ksup have identical structure.
                _gemm!('n', 't', jlen, nj, nk, -ONE, view(lnz, klpnt:length(lnz)), ksuplen, view(unz, kupnt:length(unz)), ksuplen - nk, ONE, view(lnz, jlpnt:length(lnz)), jlen)
                if  (jlen > nj)
                    _gemm!('n', 't', jlen - nj, nj, nk, -ONE, view(unz, (kupnt + nj):length(unz)), ksuplen - nk, view(lnz, klpnt:length(lnz)), ksuplen, ONE, view(unz, jupnt:length(unz)), jlen - nj)
                end
                nups = nj
                if  (klen > nj)
                    nxt = lindx[jxpnt + nj]
                end
            else
#               Sparse cmod(jsup, ksup). Determine the number of rows / columns
#               to be updated.
                nups = klen
                for i in 0:1:(klen - 1)
                    nxt = lindx[kxpnt + i]
                    if  (nxt > lj)
                        nups = i
                        break
                    end
                end  
                if (nk == 1)
#                   Updating target supernode by a trivial supernode (with one
#                   column).
                    _mmpyi!(klen, nups, view(lindx, kxpnt:length(lindx)), view(lindx, kxpnt:length(lindx)), view(lnz, klpnt:length(lnz)), view(unz, kupnt:length(unz)), xlnz, lnz, map)

                    _mmpyi!(klen - nups, nups, view(lindx, (kxpnt + nups):length(lindx)), view(lindx, kxpnt:length(lindx)), view(unz, (kupnt + nups):length(unz)), view(lnz, klpnt:length(lnz)), xunz, unz, map)
                 else
#                   kfirst ...  first index of active portion of
#                                supernode ksup (first column to be
#                                updated).
#                   klast  ...  last index of active portion of
#                                supernode ksup.
                    kfirst = lindx[kxpnt]
                    klast  = lindx[kxpnt + klen - 1]
                    inddif = map[kfirst] - map[klast]
                    if (inddif < klen)
#                       Dense cmod(jsup, ksup). ilpnt  ...  pointer to first
#                       nonzero in column kfirst.
                        ilpnt = xlnz[kfirst] + (kfirst - fj)
                        _gemm!('n', 't', klen, nups, nk, -ONE, view(lnz, klpnt:length(lnz)), ksuplen, view(unz, kupnt:length(unz)), ksuplen - nk, ONE, view(lnz, ilpnt:length(lnz)), jlen)
                        iupnt = xunz[kfirst]
                        if  (klen > nups)
                            _gemm!('n', 't', klen - nups, nups, nk, -ONE, view(unz, (kupnt + nups):length(unz)), ksuplen - nk, view(lnz, klpnt:length(lnz)), ksuplen, ONE, view(unz, iupnt:length(unz)), jlen - nj)
                        end
                    else
#                       General sparse cmod(jsup, ksup). compute cmod
#                       (jsup, ksup) update in work storage.
                        store = klen * nups
                        if  (store > tmpsiz)
                            iflag = - 2
                        end
#                       Gather indices of ksup relative to jsup.
                        _igathr!(klen, view(lindx, kxpnt:length(lindx)), map, relind)
                        _gemm!('n', 't', klen, nups, nk, -ONE, view(lnz, klpnt:length(lnz)), ksuplen, view(unz, kupnt:length(unz)), ksuplen - nk, zero(FT), vtemp, klen)
#                       Incorporate the cmod(jsup, ksup) block update into the
#                       appropriate columns of L.
                        _assmb!(klen, nups, temp, view(relind, 1:length(relind)), view(relind, 1:length(relind)), view(xlnz, fj:length(xlnz)), lnz, jlen)
                        if  (klen > nups)
                            _gemm!('n', 't', klen - nups, nups, nk, -ONE, view(unz, (kupnt + nups):length(unz)), ksuplen - nk, view(lnz, klpnt:length(lnz)), ksuplen, zero(FT), vtemp, klen - nups)
#                           incorporate the cmod(jsup, ksup) block update into
#                           the appropriate rows of u.
                            _assmb!(klen - nups, nups, temp, view(relind, 1:length(relind)), view(relind, (nups + 1):length(relind)), view(xunz, fj:length(xunz)), unz, jlen)
                        end
                    end
                end
            end
#           Link ksup into linked list of the next supernode it will update and
#           decrement ksup"s active lngth. 
            if  (klen > nups)
                nxtsup = snode[nxt]
                link[ksup] = link[nxtsup]
                link[nxtsup] = ksup
                lngth[ksup] = klen - nups
            else
                lngth[ksup] = 0
            end
#           Next updating supernode (ksup).
        end
#       Apply partial lu to the diagonal block.
        iflag = _getrf!(nj, nj, view(lnz, jlpnt:length(lnz)), jlen, view(ipvt, fj:length(ipvt)))
        if  (iflag != 0)
            iflag = - 1
        end
#       Update columns.
        _trsm!('r', 'u', 'n', 'n', jlen - nj, nj, ONE, view(lnz, jlpnt:length(lnz)), jlen, view(lnz, (jlpnt + nj):length(lnz)), jlen)
#       Apply permutation to rows.
        if  (jlen > nj)
            _luswap!(jlen - nj, nj, view(unz, jupnt:length(unz)), jlen - nj, view(ipvt, fj:length(ipvt)))
#           Update rows.
            _trsm!('r', 'l', 't', 'u', jlen - nj, nj, ONE, view(lnz, jlpnt:length(lnz)), jlen, view(unz, jupnt:length(unz)), jlen - nj)
        end
#       Insert jsup into linked list of first supernode it will update.
        if  (jlen > nj)
            nxt = lindx[jxpnt + nj]
            nxtsup = snode[nxt]
            link[jsup] = link[nxtsup]
            link[nxtsup] = jsup
            lngth[jsup] = jlen - nj
        else
            lngth[jsup] = 0
        end
    end

    return iflag
end


 #     Given the L U factorization of a sparse structurally symmetric
 #     positive definite matrix, this subroutine performs the
 #     triangular solution.  It uses output from LUFactor.
 # Input parameters:
 #     nsuper - number of supernodes.
 #     xsuper - supernode partition.
 #     (xlindx, lindx) - row indices for each supernode.
 #     (xlnz, lnz) - Cholesky factor.
 # Updated parameters:
 #     rhs - on input, contains the right hand side.  on
 #                          output, contains the solution.
function _lulsolve!(nsuper::IT, xsuper::Vector{IT}, xlindx::Vector{IT}, lindx::Vector{IT}, xlnz::Vector{IT}, lnz::Vector{FT}, ipiv::Vector{IT}, rhs::Vector{FT}) where {IT, FT}
     # integer :: nsuper
     # integer :: lindx(*), ipiv(*), xsuper(*), xlindx(*), xlnz(*)
     # real(double) :: lnz(*), rhs(*)

     # integer :: fj, isub, j, jj, jlen, jlpnt, jsup, jxpnt, nj
     # integer :: length, maxlength
     # real(double), dimension(:), allocatable :: temp

# -  -  -  -  -  -  -  -  -  -
#    constants.
# -  -  -  -  -  -  -  -  -  -
    ONE = one(FT)
    ZERO = zero(FT)

    if  (nsuper <= 0)  return false; end

    lngth = 0; maxlngth = 0
    for j in 1:nsuper
        lngth = xlindx[j + 1] - xlindx[j]
        maxlngth = max(lngth, maxlngth)
    end

    temp = zeros(FT,maxlngth)

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#    forward substitution ...
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    for jsup in 1:nsuper

        fj    = xsuper[jsup]
        nj    = xsuper[jsup + 1] - fj
        jlen  = xlnz[fj + 1] - xlnz[fj]
        jxpnt = xlindx[jsup]
        jlpnt = xlnz[fj]

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#       pivot rows of RHS to match pivoting of L and U.
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        _laswp!(view(rhs, fj:length(rhs)), nj, 1, nj, view(ipiv, fj:length(ipiv)))

        _trsm!('l', 'l', 'n', 'u', nj, 1, ONE, view(lnz, jlpnt:length(lnz)), jlen, view(rhs, fj:length(rhs)), nj)

        _gemv!('n', jlen - nj, nj, -ONE, view(lnz, (jlpnt + nj):length(lnz)), jlen, view(rhs, fj:length(rhs)), ZERO, view(temp, 1:length(temp)))

        jj = jxpnt + nj - 1
        for  j in 1:(jlen - nj)
            jj = jj + 1
            isub = lindx[jj]
            rhs[isub] += temp[j]
            temp[j] = ZERO
        end
    end
    return true
end

function _luusolve!(n::IT, nsuper::IT, xsuper::Vector{IT}, xlindx::Vector{IT}, lindx::Vector{IT}, xlnz::Vector{IT}, lnz::Vector{FT}, xunz::Vector{IT}, unz::Vector{FT}, rhs::Vector{FT}) where {IT, FT}
# integer :: nsuper, n
# integer :: lindx(*), xsuper(*), xlindx(*), xlnz(*), xunz(*)
# real(double) :: lnz(*), rhs(*), unz(*)

# integer :: fj, isub, j, jj, jlen, jlpnt, jsup, jupnt, jxpnt, nj
# integer :: length, maxlength
# real(double), dimension(:), allocatable :: temp

    ONE = one(FT)
    ZERO = zero(FT)

    if  (nsuper <= 0)  return; end

    lngth = 0; maxlngth = 0
    for j in 1:nsuper
        lngth = xlindx[j + 1] - xlindx[j]
        maxlngth = max(lngth, maxlngth)
    end

    temp = zeros(FT, maxlngth)

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#    backward substitution ...
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    for jsup in nsuper:-1:1

        fj    = xsuper[jsup]
        nj    = xsuper[jsup + 1] - fj
        jlen  = xlnz[fj + 1] - xlnz[fj]
        jxpnt = xlindx[jsup]
        jlpnt = xlnz[fj]
        jupnt = xunz[fj]

        jj = jxpnt + nj - 1
        for j in 1:(jlen - nj)
            jj = jj + 1
            isub = lindx[jj]
            if (isub > n)
                error("Index $(isub) out of bounds in rhs.")
                return false
            end
            temp[j] = rhs[isub]
        end  
        if (jlen > nj)
            _gemv!('t', jlen - nj, nj, -ONE, view(unz, jupnt:length(unz)), jlen - nj, view(temp, 1:length(temp)), ONE, view(rhs, fj:length(rhs)))
        end

        _trsm!('l', 'u', 'n', 'n', nj, 1, ONE, view(lnz, jlpnt:length(lnz)), jlen, view(rhs, fj:length(rhs)), nj)

    end
    return true
end

end  # module



