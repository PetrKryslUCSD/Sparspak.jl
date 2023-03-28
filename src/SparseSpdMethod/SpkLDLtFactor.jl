module SpkLDLtFactor
#
#  LDLtFactor ..... BLOCK GENERAL SPARSE CHOL   ^^^^^^
# *
# *
# """
#  purpose:
#      this subroutine computes an LDL^t factorization of a sparse
#      matrix with symmetric structure.  the computation is
#      organized around kernels that perform supernode - to - supernode
#      updates.  level - 3 blas routines used whenever possible.
#      each supernode is a dense trapezoidal submatrix stored in
#      a rectangular array.
##
#  input parameters:
#      nsuper - number of supernode.
#      xsuper - supernode partition.
#      snode - maps each column to the supernode containing it.
#      (xlindx, lindx) - row indices for each supernode (including
#                          the diagonal elements).
#      (xnz, lnz) - on input, contains matrix to be factored.
##
#  output parameters:
#      lnz - on output, contains cholesky factor.
#      iflag - error flag.
#                              0: successful factorization.
#- 1: zero diagonal encountered,
#                                 matrix is (numerically) singular
##
#  local variables:
#      link - a linked list that stores the supernode
#                          modification list.  if link(jsup) = k,
#                          supernodes k, link(k), link(link(k)), ... all
#                          modify supernode jsup.
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
#                          length(ksup), and only stores the elements
#                          from map coresponding to the union of row indxes
#                          of jsup and ksup.
#      temp - real vector for accumulating updates.  must
#                          accomodate all columns of a supernode.
#      temp2 - real vector for storing the DL^t computation.
#
# *
# """
function ldltfactor!(n, nsuper, xsuper, snode , xlindx , lindx, xlnz, lnz)
    @assert length(xsuper) == (nsuper + 1)
    @assert length(lngth) == (nsuper)
    @assert length(snode) == n
    @assert length(xlindx) == (nsuper + 1)
    @assert length(link) == (nsuper)
    @assert length(xlnz) == (n + 1)

    iflag = 0;   width = 0;   maxwidth = 0; tmpsiz = 0

#       initialize empty row lists in link(*).
    for i  in  1: nsuper
        link[i] = 0;   length[i] = xlindx[i + 1] - xlindx[i]
        width = xsuper[i + 1] - xsuper[i]
        need = width * (xlnz[xsuper[i] + 1] - xlnz[xsuper[i]])
        if (need > tmpsiz) tmpsiz = need; end
        maxwidth = max(maxwidth, width)
    end

    map = fill(zero(IT), n)
    relind = fill(zero(IT), n)
    diag = fill(zero(IT), nmaxwidth)
    temp = fill(zero(FT), tmpsiz)
    temp2 = fill(zero(FT), maxwidth * maxwidth)

#      for each supernode jsup ...
       for jj  in  1: nsuper
          jsup = jj
#         [f / l]j ...  first / last column of jsup.
#         nj     ...  number of columns in jsup.
#         jlen   ...  length of jsup.
#         jxpnt  ...  pointer to index of first nonzero in
#                     jsup.
#         jlpnt  ...  pointer to location of first nonzero
#                     in column fj.
          fj    = xsuper[jsup]; lj = xsuper[jsup + 1] - 1
          nj    = lj - fj + 1
          jlen  = xlnz[fj + 1] - xlnz[fj]
          jxpnt = xlindx[jsup]
          jlpnt = xlnz[fj]
#         set up map(*) to map the entries in update columns
#         to their corresponding positions in updated columns,
#         relative to the bottom of each updated column.
          _ldindx!(jlen, view(lindx, jxpnt:length(lindx)), map)
#         for every supernode ksup in row(jsup) ...
          while (true)
#            take out the first item in the list.  MUST BE DONE
#            atomically
            ksup = link[jsup]
            if (ksup != 0)
#                  remove the head of the list.
                link[jsup] = link[ksup]
                link[ksup] = 0
            end
            if (ksup == 0) break; end
#            get info about the cmod(jsup, ksup) update.
#
#            [f / l]k   ...  first / last column of ksup.
#            nk       ...  number of columns in ksup.
#            ksuplen  ...  length of ksup.
#            klen     ...  length of active portion of ksup.
#            kxpnt    ...  pointer to index of first nonzero in
#                          active portion of ksup.
#            klpnt    ...  pointer to location of first nonzero in
#                          active portion of column fk.
             fk  = xsuper[ksup]
             lk  = xsuper[ksup + 1] - 1
             nk = lk - fk + 1
             ksuplen  = xlnz[fk + 1] - xlnz[fk]
             klen   = lngth[ksup]
             kxpnt  = xlindx[ksup + 1] - klen
             klpnt = xlnz[fk + 1] - klen

             _loaddiag!(view(lnz, xlnz[fk]:length(lnz)), ksuplen, nk, diag)
#            perform cmod(jsup, ksup), with special cases
#            handled differently.
#            nups  ...  number of rows / columns updated.
             if (klen == jlen)
#               dense cmod(jsup, ksup).
#               jsup and ksup have identical structure.
                _matrixdiagmm!(view(lnz, klpnt:length(lnz)), nj, nk, ksuplen, diag, temp2)
                _gemm!('n', 't', jlen, nj, nk, - ONE, view(lnz, klpnt:length(lnz)), ksuplen, temp2, nj, one, view(lnz, jlpnt:length(lnz)), jlen)
                nups = nj
                if  (klen > nj)
                    nxt = lindx[jxpnt + nj]
                end
            else
#               sparse cmod(jsup, ksup).
#               determine the number of rows / columns
#               to be updated.
                nups = klen
                for i  in  0: (klen - 1)
                    nxt = lindx[kxpnt + i]
                    if  (nxt > lj)
                        nups = i
                        break
                    end
                end
                if (nk == 1)
#                  updating target supernode by a trivial
#                  supernode (with one column).
                    _mmpyi!(klen, nups, view(lindx, kxpnt:length(lindx)), view(lindx, kxpnt:length(lindx)), view(lnz, klpnt:length(lnz)), view(lnz, klpnt:length(lnz)), xlnz, lnz, map, view(lnz, xlnz[fk]:length(lnz)))

                else
#                  kfirst ...  first index of active portion of
#                              supernode Ksup (first column
#                              to be updated).
#                  klast  ...  last index of active portion of
#                              supernode Ksup.
                    kfirst = lindx[kxpnt]
                    klast  = lindx[kxpnt + klen - 1]
                    inddif = map[kfirst] - map[klast]
                    if  (inddif < klen)
#                     dense cmod(Jsup, Ksup).
#
#                     ilpnt  ...  pointer to first nonzero in
#                                 column kfirst.
                        ilpnt = xlnz[kfirst] + (kfirst - fj)
                        width = nups
                        _matrixdiagmm!(view(lnz, klpnt:length(lnz)), width, nk, ksuplen, diag, temp2)

                        _gemm!('n', 't', klen, nups, nk, - ONE, view(lnz, klpnt:length(lnz)), ksuplen, temp2, width, ONE, view(lnz, ilpnt:length(lnz)), jlen)
                    else
#                     general sparse cmod(Jsup, Ksup).
#                     compute cmod(Jsup, Ksup) update
#                     in work storage.
                        store = klen * nups
                        if  (store > tmpsiz)
                            iflag = - 2
                        end
#                     gather indices of ksup relative to jsup.
                        __igathr!(klen, lindx(kxpnt), map, relind)
                        width = nups
                        _matrixdiagmm!(view(lnz, klpnt:length(lnz)), width, nk, ksuplen, diag, temp2)

                        _gemm!('n', 't', klen, nups, nk, - ONE, view(lnz, klpnt:length(lnz)), ksuplen, temp2, width, 0.0, temp, klen)
#                     incorporate the cmod(Jsup, Ksup) block
#                     update into the appropriate columns of l.
                        _assmb!(klen, nups, temp, view(relind, 1:length(relind)), view(relind, 1:length(relind)), view(xlnz, fj:length(xlnz)), lnz, jlen)
                    end
                end
            end
#            link Ksup into linked list of the next
#            supernode
#            it will update and decrement ksup"s active
#            length.
            if (klen > nups)
                nxtsup = snode[nxt]
                link[ksup] = link[nxtsup]
                link[nxtsup] = ksup
                length[ksup] = klen - nups
            else
                length[ksup] = 0
            end
        end
#         Apply partial LDLt to the supernode
        _pchole!(view(lnz, xlnz[fj]:length(lnz)), nj, jlen)
#         insert Jsup into linked list of first
#         supernode it will update.
        if  (jlen > nj)
            nxt = lindx[jxpnt + nj]
            nxtsup = snode[nxt]
            link[jsup] = link[nxtsup]
            link[nxtsup] = jsup
            lngth[jsup] = jlen - nj
        else
            lngth[jsup] = 0
        end
#
#         decrement the counter on all depending supernodes
#
    end

    return iflag
end

#
#
# *     LDLtSolve ... block triangular solutions ^^^^^^^^^^ *
#
# """   Purpose:
#      Given the L D L^T factorization of a sparse symmetric
#      positive definite matrix, this subroutine performs the
#      triangular solution.  It uses output from LDLtFactor.
#  Input parameters:
#      nsuper - number of supernodes.
#      xsuper - supernode partition.
#      (xlindx, lindx) - row indices for each supernode.
#      (xlnz, lnz) - Cholesky factor.
#  Updated parameters:
#      rhs - on input, contains the right hand side.  on
#                          output, contains the solution.
"""
"""
function _ldltsolve!(nsuper, xsuper, xlindx, lindx, xlnz, lnz, rhs)
#    forward substitution ...
     for jsup  in  1: nsuper
        fjcol = xsuper[jsup];   ljcol = xsuper[jsup + 1] - 1;
        fsub = xlindx[jsup];   lsub = xlindx[jsup + 1] - 1;
        for jcol  in  fjcol: ljcol
            ofst = jcol - fjcol
            nzf = xlnz[jcol] + 1 + ofst;   nzl = xlnz[jcol + 1] - 1;
            fsub = fsub + 1;   t = rhs[jcol];
            rhs[lindx[fsub:lsub]] = rhs[lindx[fsub:lsub]] - t * lnz[nzf:nzl]
        end
    end
#    backward substitution ...
    for jsup  in  nsuper:-1:1
        fjcol = xsuper[jsup];   ljcol = xsuper[jsup + 1] - 1;
        fsub = xlindx[jsup] + ljcol - fjcol;
        for jcol  in  ljcol:-1:fjcol
            ofst = jcol - fjcol
            nzf = xlnz[jcol] + ofst;   m = xlnz[jcol + 1] - nzf - 1;
            t = rhs[jcol] / lnz[nzf]
            for k  in  1: m
                t = t - lnz[nzf + k] * rhs[lindx[fsub + k]]
            end
            rhs[jcol] = t;   fsub = fsub - 1
        end
    end
    return true
end

#       This routine will perform matrix multiplication by a diagonal
#       matrix.  The diagonal matrix is stored as a vector, and will
#       scale the rows of the output matrix
#
#   INPUT PARAMETERS -
#       lnz - pointer to the first non - diagonal, non - zero in the
#                 supernode
#       klen - length of the section to copy into temp, and the length
#                 of temp
#       nk - width of the supernode and temp
#       ksuplen - length of each column of the supernode
#       diag - a vector representation of a diagonal matrix
#
#   UPDATED PARAMETERS -
#       temp - On output, temp is the result of the multiplication

@inline function _twod_access(nr, i, j)
    (j-1)*nr + i
end

function _matrixdiagmm!(lnz, klen, nk, ksuplen, diag, temp)
    # The two dimensional arrays are being passed in as 1d views. The two
    # dimensional access needs to be faked.
    # real(double) :: lnz(ksuplen,nk), temp(klen,nk), diag(nk)
    for  j  in  1: nk
        diagonal = diag[j]
        for r in 1:klen
            temp[_twod_access(klen, r, j)] = diagonal*lnz[_twod_access(klen, r, j)]
        end
    end
end


function _loaddiag!(lnz, ksuplen, nk, diag)
    # real(double) :: lnz(ksuplen,nk), diag(*)
    for j in 1:nk
        diag[j] = lnz[_twod_access(ksuplen, j, j)]
    end
end


# !       this routine will apply the partial cholesky factorization to
# !       a supernode through a left looking algorithm.
# !       This algorithm assumes that supernodes are rectangular, rather
# !       than trapezoidal.
# !
# !   INPUT PARAMETERS -
# !       lnz     - on input, contains the matrix starting at the
# !                 supernode.
# !        nj     - number of column in this supernode
# !        lda    - the number of rows in the supernode
# !
function _pchole!(lnz, nj, lda)
# real(double) :: lnz(lda, *) the 2d access needs to be faked
    for colnum  in  1: nj
#       factor the diagonal block.
        for i = 1: colnum - 1
            for r in colnum:nj
                lnz[_twod_access(lda, r, colnum)] +=  - lnz[_twod_access(lda, colnum, i)] * lnz[_twod_access(lda, i, i)] * lnz[_twod_access(lda, r, i)]
            end
            diag = lnz[_twod_access(lda, colnum, colnum)]
            for r in colnum + 1:nj
                lnz[_twod_access(lda, r, colnum)] /= diag
            end
        end
    end

# -  -  -  -  -
#    factor the lower block.
#
    _trsm!('r', 'l', 't', 'u', lda - nj, nj, ONE, lnz, lda, view(lnz, nj + 1:length(lnz)), lda)

# -
#    scale back each column by the diagonal.
# -
    for colnum  in  1: nj
        diag = lnz[_twod_access(lda, colnum, colnum)]
        for r in nj + 1:lda
            lnz[_twod_access(lda, r, colnum)] /= diag
        end
    end
end
#
end # module spkldltfactor


