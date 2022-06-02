module SpkLUFactor

using ..SpkSpdMMops: mmpyi, assmb, luswap, ldindx, igathr, dgemm!, dtrsm!, dgetrf!, dlaswp!, dgemv!

"""  purpose:
     this subroutine computes an LU factorization of a sparse
     matrix with symmetric structure.  the computation is
     organized around kernels that perform supernode - to - supernode
     updates.  level - 3 blas routines used whenever possible.
     each supernode is a dense trapezoidal submatrix stored in
     a rectangular array.
#
 input parameters:
     nsuper - number of supernode.
     xsuper - supernode partition.
     snode - maps each column to the supernode containing it.
     (xlindx, lindx) - row indices for each supernode (including
                         the diagonal elements).
     (xlnz, lnz) - on input, contains the lower half of the matrix
                         to be factored and the diagonal block.
     (xunz, unz) - on input, contains the upper half of the matrix
                         to be factored.
#
 output parameters:
     lnz - on output, contains the lower half of the
                         cholesky factor.
     unz - on output, contains the upper half of the
                         cholesky factor.
     iflag - error flag.
                             0: successful factorization.
- 1: nonpositive diagonal encountered,
                                matrix is not positive definite.
- 2: insufficient working storage
                                [temp(*)].
     ipvt - row pivoting information.
#
 local variables:
     link - a linked list that stores the supernode
                         that are ready to modify jsup.
                         if link(jsup) = k,  supernodes k,
                         link(k), link(link(k)), ... all modify
                         supernode jsup.
     length - length of the active portion of each supernode.
                         ie. the length that hasn"t effected a diagonal
                         block yet.
     map - vector of size n into which the global
                         indices are scattered.  This array holds the
                         distance of row k from the bottom of the
                         supernode if it were compressed.
     relind - maps locations in the updating columns to
                         the corresponding locations in the updated
                         columns.  (relind is gathered from map).
                         more specifically, this is an array of length
                         lenght(ksup), and only stores the elements
                         from map coresponding to union of row indexes
                         of jsup and ksup.
     temp - real vector for accumulating updates.  must
                          accomodate all columns of a supernode.
"""
function lufactor(n::IT, nsuper::IT, xsuper::Vector{IT}, snode::Vector{IT}, xlindx::Vector{IT}, lindx::Vector{IT}, xlnz::Vector{IT}, lnz::Vector{FT}, xunz::Vector{IT}, unz::Vector{FT}, ipvt::Vector{IT}) where {IT, FT}
# integer :: n, nsuper
# integer :: xlindx(nsuper + 1), xlnz(n + 1), xunz(n + 1)
# integer :: lindx(*), lngth(nsuper), link(nsuper), snode(n)
# integer :: xsuper(nsuper + 1), ipvt(n)
# integer :: iflag, tmpsiz
# real(double) :: lnz(*), unz(*)
# real(double), dimension(:), allocatable :: temp
# integer :: effectedsn, width
# integer, dimension(:), allocatable :: map, relind

# integer :: fj, fk, lj, lk, nj, nk, i, j, jj, ilpnt, iupnt, inddif
# integer :: jlen, jlpnt, jsup, jupnt, jxpnt, kfirst, klast
# integer :: klen, klpnt, ksup, ksuplen, kupnt, kxpnt
# integer :: nups, nxt, nxtsup, store, nxksup
# integer :: need

    @assert length(xsuper) == (nsuper + 1)
    @assert length(snode) == n
    @assert length(xlindx) == (nsuper + 1)
    @assert length(xlnz) == (n + 1)
    @assert length(xunz) == (n + 1)
    @assert length(ipvt) == n

    ONE = FT(1.0)
    
    link = fill(zero(IT), nsuper)
    lngth = fill(zero(IT), nsuper)
#
    iflag = zero(IT);   tmpsiz = zero(IT)

# -------------
#      initialization
# -------------
    for i in 1:nsuper
        link[i] = 0
        lngth[i] = xlindx[i + 1] - xlindx[i]
        width = xsuper[i + 1] - xsuper[i]
        need = width * (xlnz[xsuper[i] + 1] - xlnz[xsuper[i]])
        if (need > tmpsiz) tmpsiz = need; end
    end

    map = fill(zero(IT), n)
    relind = fill(zero(IT), n)
    temp = fill(zero(FT), tmpsiz)
    vtemp = view(temp, 1:length(temp))

# --------------------------
#      for each supernode jsup ...
# --------------------------
    for jj in 1:nsuper
        jsup = jj

#          fj     ...  first row / column of jsup.
#          lj     ...  last row / column of jsup.
#          nj     ...  number of rows / columns in jsup.
#          jlen   ...  lngth of jsup.
#          jxpnt  ...  pointer to index of first nonzero in
#                      jsup.
#          jlpnt  ...  pointer to location of first nonzero
#                      in column fj.
#          jupnt  ...  pointer to location of first nonzero
#                      in row fj.

        fj    = xsuper[jsup]
        lj    = xsuper[jsup + 1] - 1
        nj    = lj - fj + 1
        jlen  = xlnz[fj + 1] - xlnz[fj]
        jxpnt = xlindx[jsup]
        jlpnt = xlnz[fj]
        jupnt = xunz[fj]

# 
#          set up map(*) to map the entries in update columns
#          to their corresponding positions in updated columns,
#          relative to the bottom of each updated column.
# 
        ldindx(jlen, view(lindx, jxpnt:length(lindx)), map)

#          for every supernode ksup in row(jsup) ...

        while (true)

#             wait for something to appear in the list, or
#             stop if there can"t be anything more

#             take out the first item in the list.  MUST BE DONE
#             atomically#

            ksup = link[jsup]
            if (ksup != 0)
#                   remove the head of the list.
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

#             perform cmod(jsup, ksup), with special cases
#             handled differently.
#
#             each cmod is broken up into two stages:
#                 first perform cmod on the columns,
#                  perform cmod on the rows.
#
#             nups  ...  number of rows / columns updated.

            if  (klen == jlen)

#                dense cmod(jsup, ksup).
#                jsup and ksup have identical structure.

                dgemm!('n', 't', jlen, nj, nk, -ONE, view(lnz, klpnt:length(lnz)), ksuplen, view(unz, kupnt:length(unz)), ksuplen - nk, ONE, view(lnz, jlpnt:length(lnz)), jlen)

                if  (jlen > nj)
                    dgemm!('n', 't', jlen - nj, nj, nk, -ONE, view(unz, (kupnt + nj):length(unz)), ksuplen - nk, view(lnz, klpnt:length(lnz)), ksuplen, ONE, view(unz, jupnt:length(unz)), jlen - nj)
                end

                nups = nj
                if  (klen > nj)
                    nxt = lindx[jxpnt + nj]
                end

            else
# 
#                sparse cmod(jsup, ksup).
# 
# 
#                determine the number of rows / columns
#                to be updated.
# 
                nups = klen
                for i in 0:1:(klen - 1)
                    nxt = lindx[kxpnt + i]
                    if  (nxt > lj)
                        nups = i
                        break
                    end
                end  

                if (nk == 1)
# 
#                   updating target supernode by a trivial
#                   supernode (with one column).
# 
                    mmpyi(klen, nups, view(lindx, kxpnt:length(lindx)), view(lindx, kxpnt:length(lindx)), view(lnz, klpnt:length(lnz)), view(unz, kupnt:length(unz)), xlnz, lnz, map)

                    mmpyi(klen - nups, nups, view(lindx, (kxpnt + nups):length(lindx)), view(lindx, kxpnt:length(lindx)), view(unz, (kupnt + nups):length(unz)), view(lnz, klpnt:length(lnz)), xunz, unz, map)

                 else
# 
#                    kfirst ...  first index of active portion of
#                                supernode ksup (first column to be
#                                updated).
#                    klast  ...  last index of active portion of
#                                supernode ksup.
# 
                    kfirst = lindx[kxpnt]
                    klast  = lindx[kxpnt + klen - 1]
                    inddif = map[kfirst] - map[klast]

                    if (inddif < klen)
# 
#                      dense cmod(jsup, ksup).
#
#                      ilpnt  ...  pointer to first nonzero in
#                                  column kfirst.
# 

                        ilpnt = xlnz[kfirst] + (kfirst - fj)
                        dgemm!('n', 't', klen, nups, nk, -ONE, view(lnz, klpnt:length(lnz)), ksuplen, view(unz, kupnt:length(unz)), ksuplen - nk, ONE, view(lnz, ilpnt:length(lnz)), jlen)

                        iupnt = xunz[kfirst]
                        if  (klen > nups)
                            dgemm!('n', 't', klen - nups, nups, nk, -ONE, view(unz, (kupnt + nups):length(unz)), ksuplen - nk, view(lnz, klpnt:length(lnz)), ksuplen, ONE, view(unz, iupnt:length(unz)), jlen - nj)
                        end
                    else
# 
#                      general sparse cmod(jsup, ksup).
#                      compute cmod(jsup, ksup) update
#                      in work storage.
# 
                        store = klen * nups
                        if  (store > tmpsiz)
                            iflag = - 2
                        end

# 
#                      gather indices of ksup relative
#                      to jsup.
# 
                        igathr(klen, view(lindx, kxpnt:length(lindx)), map, relind)

                        dgemm!('n', 't', klen, nups, nk, -ONE, view(lnz, klpnt:length(lnz)), ksuplen, view(unz, kupnt:length(unz)), ksuplen - nk, zero(FT), vtemp, klen)

# 
#                      incorporate the cmod(jsup, ksup) block
#                      update into the appropriate columns of l.
# 
                        assmb(klen, nups, temp, view(relind, 1:length(relind)), view(relind, 1:length(relind)), view(xlnz, fj:length(xlnz)), lnz, jlen)

                        if  (klen > nups)

                            dgemm!('n', 't', klen - nups, nups, nk, -ONE, view(unz, (kupnt + nups):length(unz)), ksuplen - nk, view(lnz, klpnt:length(lnz)), ksuplen, zero(FT), vtemp, klen - nups)

# -
#                         incorporate the cmod(jsup, ksup) block
#                         update into the appropriate rows of u.
# 
                            assmb(klen - nups, nups, temp, view(relind, 1:length(relind)), view(relind, (nups + 1):length(relind)), view(xunz, fj:length(xunz)), unz, jlen)
                        end
                    end
                end
            end

# 
#             link ksup into linked list of the next supernode
#             it will update and decrement ksup"s active
#             lngth.
# 
            if  (klen > nups)
                nxtsup = snode[nxt]
                link[ksup] = link[nxtsup]
                link[nxtsup] = ksup
                lngth[ksup] = klen - nups
            else
                lngth[ksup] = 0
            end

# ----------------------------
#             next updating supernode (ksup).
# 
        end

# 
#          apply partial lu to the diagonal block.
# 
        
        iflag = dgetrf!(nj, nj, view(lnz, jlpnt:length(lnz)), jlen, view(ipvt, fj:length(ipvt)))

        if  (iflag != 0)
            iflag = - 1
        end

#          update columns.

        dtrsm!('r', 'u', 'n', 'n', jlen - nj, nj, ONE, view(lnz, jlpnt:length(lnz)), jlen, view(lnz, (jlpnt + nj):length(lnz)), jlen)

#          apply permutation to rows.

        if  (jlen > nj)
            luswap(jlen - nj, nj, view(unz, jupnt:length(unz)), jlen - nj, view(ipvt, fj:length(ipvt)))

#             update rows.

            dtrsm!('r', 'l', 't', 'u', jlen - nj, nj, ONE, view(lnz, jlpnt:length(lnz)), jlen, view(unz, jupnt:length(unz)), jlen - nj)
        end

#          insert jsup into linked list of first supernode
#          it will update.

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

"""  Purpose:
     Given the L U factorization of a sparse structurally symmetric
     positive definite matrix, this subroutine performs the
     triangular solution.  It uses output from LUFactor.
 Input parameters:
     nsuper - number of supernodes.
     xsuper - supernode partition.
     (xlindx, lindx) - row indices for each supernode.
     (xlnz, lnz) - Cholesky factor.
 Updated parameters:
     rhs - on input, contains the right hand side.  on
                          output, contains the solution.
"""
function lulsolve(nsuper::IT, xsuper::Vector{IT}, xlindx::Vector{IT}, lindx::Vector{IT}, xlnz::Vector{IT}, lnz::Vector{FT}, ipiv::Vector{IT}, rhs::Vector{FT}) where {IT, FT}
     # integer :: nsuper
     # integer :: lindx(*), ipiv(*), xsuper(*), xlindx(*), xlnz(*)
     # real(double) :: lnz(*), rhs(*)

     # integer :: fj, isub, j, jj, jlen, jlpnt, jsup, jxpnt, nj
     # integer :: length, maxlength
     # real(double), dimension(:), allocatable :: temp

# -  -  -  -  -  -  -  -  -  -
#    constants.
# -  -  -  -  -  -  -  -  -  -
    ONE = FT(1.0)
    ZERO = FT(0.0)

    if  (nsuper <= 0)  return false; end

    lngth = 0; maxlngth = 0
    for j in 1:nsuper
        lngth = xlindx[j + 1] - xlindx[j]
        maxlngth = max(lngth, maxlngth)
    end

    temp = fill(zero(FT), maxlngth)

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
        dlaswp!(view(rhs, fj:length(rhs)), nj, 1, nj, view(ipiv, fj:length(ipiv)))

        dtrsm!('l', 'l', 'n', 'u', nj, 1, ONE, view(lnz, jlpnt:length(lnz)), jlen, view(rhs, fj:length(rhs)), nj)

        dgemv!('n', jlen - nj, nj, -ONE, view(lnz, (jlpnt + nj):length(lnz)), jlen, view(rhs, fj:length(rhs)), ZERO, view(temp, 1:length(temp)))

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
"""
"""
function luusolve(n::IT, nsuper::IT, xsuper::Vector{IT}, xlindx::Vector{IT}, lindx::Vector{IT}, xlnz::Vector{IT}, lnz::Vector{FT}, xunz::Vector{IT}, unz::Vector{FT}, rhs::Vector{FT}) where {IT, FT}
     # integer :: nsuper, n
     # integer :: lindx(*), xsuper(*), xlindx(*), xlnz(*), xunz(*)
     # real(double) :: lnz(*), rhs(*), unz(*)

     # integer :: fj, isub, j, jj, jlen, jlpnt, jsup, jupnt, jxpnt, nj
     # integer :: length, maxlength
     # real(double), dimension(:), allocatable :: temp

    ONE = FT(1.0)
    ZERO = FT(0.0)

     if  (nsuper <= 0)  return; end

     lngth = 0; maxlngth = 0
     for j in 1:nsuper
        lngth = xlindx[j + 1] - xlindx[j]
        maxlngth = max(lngth, maxlngth)
     end

     temp = fill(zero(FT), maxlngth)

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
              @error "$(@__FILE__): $(isub) index out of bounds in rhs"
              return false
          end
           temp[j] = rhs[isub]
        end  
        if (jlen > nj)
           dgemv!('t', jlen - nj, nj, -ONE, view(unz, jupnt:length(unz)), jlen - nj, view(temp, 1:length(temp)), ONE, view(rhs, fj:length(rhs)))
        end

        dtrsm!('l', 'u', 'n', 'n', nj, 1, ONE, view(lnz, jlpnt:length(lnz)), jlen, view(rhs, fj:length(rhs)), nj)

     end
return true
   end

end  # module



