module SpkLUFactor

using ..SpkSpdMMops: mmpyi, assmb, luswap, ldindx, igathr

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
function lufactor(n, nsuper, xsuper, snode, xlindx, lindx, xlnz, lnz, xunz, unz, ipvt)
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
#
    IT = eltype(xlindx)
    FT = eltype(lnz)

    iflag = zero(IT);   tmpsiz = zero(IT)

# --------------
#      initialization
# --------------
    for i in 1:nsuper
        link[i] = 0
        lngth[i] = xlindx[i + 1] - xlindx[i]
        width = xsuper[i + 1] - xsuper[i]
        need = width * (xlnz[xsuper[i] + 1] - xlnz[xsuper[i]])
        if (need > tmpsiz) tmpsiz = need; end
    end

    map = fill(zero(IT), n)
    relind = fill(zero(IT), n)
    temp = fill(zero(IT), tmpsiz)

# ---------------------------
#      for each supernode jsup ...
# ---------------------------
    for jj in 1:nsuper
        jsup = jj

# -
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
# --
        fj    = xsuper[jsup]
        lj    = xsuper[jsup + 1] - 1
        nj    = lj - fj + 1
        jlen  = xlnz[fj + 1] - xlnz[fj]
        jxpnt = xlindx[jsup]
        jlpnt = xlnz[fj]
        jupnt = xunz[fj]

# -
#          set up map(*) to map the entries in update columns
#          to their corresponding positions in updated columns,
#          relative to the bottom of each updated column.
# -
        ldindx(jlen, lindx[jxpnt], map)

# ---
#          for every supernode ksup in row(jsup) ...
# --
        while (true)
# --
#             wait for something to appear in the list, or
#             stop if there can"t be anything more
# --

# --
#             take out the first item in the list.  MUST BE DONE
#             atomically#
# --
            ksup = link[jsup]
            if (ksup != 0)
# ----------------------------
#                   remove the head of the list.
# ---------------------------
                link[jsup] = link[ksup]
                link[ksup] = 0
            end

            if (ksup == 0) break; end
# ---
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
# -- 
            fk  = xsuper[ksup]
            lk  = xsuper[ksup + 1] - 1
            nk = lk - fk + 1
            ksuplen  = xlnz[fk + 1] - xlnz[fk]
            klen   = lngth[ksup]
            kxpnt  = xlindx[ksup + 1] - klen
            klpnt = xlnz[fk + 1] - klen
            kupnt = xunz[fk + 1] - klen

# -
#             perform cmod(jsup, ksup), with special cases
#             handled differently.
#
#             each cmod is broken up into two stages:
#                 first perform cmod on the columns,
#                  perform cmod on the rows.
#
#             nups  ...  number of rows / columns updated.
# 
            if  (klen == jlen)

# 
#                dense cmod(jsup, ksup).
#                jsup and ksup have identical structure.
# 
                dgemm("n", "t", jlen, nj, nk, -one(FT), lnz[klpnt], ksuplen, unz[kupnt], ksuplen - nk, one, lnz[jlpnt], jlen)

                if  (jlen > nj)
                    dgemm("n", "t", jlen - nj, nj, nk, -one(FT), unz[kupnt + nj], ksuplen - nk, lnz[klpnt], ksuplen, one(FT), unz[jupnt], jlen - nj)
                end

                nups = nj
                if  (klen > nj)
                    nxt = lindx[jxpnt + nj]
                end

            else
# ---------------------------
#                sparse cmod(jsup, ksup).
# ---------------------------
# ------------------------------------
#                determine the number of rows / columns
#                to be updated.
# ------------------------------------
                nups = klen
                for i in 0:1:(klen - 1)
                    nxt = lindx[kxpnt + i]
                    if  (nxt > lj)
                        nups = i
                        break
                    end
                end  

                if (nk == 1)
# --------------------------------------
#                   updating target supernode by a trivial
#                   supernode (with one column).
# --------------------------------------
                    mmpyi(klen, nups, lindx[kxpnt], lindx[kxpnt], lnz[klpnt], unz[kupnt], xlnz, lnz, map)

                    mmpyi(klen - nups, nups, lindx(kxpnt + nups), lindx[kxpnt], unz[kupnt + nups], lnz[klpnt], xunz, unz, map)

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
# ---------------------------------------
#                      dense cmod(jsup, ksup).
#
#                      ilpnt  ...  pointer to first nonzero in
#                                  column kfirst.
# ---------------------------------------

                        ilpnt = xlnz[kfirst] + [kfirst - fj]
                        dgemm("n", "t", klen, nups, nk, -one(FT), lnz[klpnt], ksuplen, unz[kupnt], ksuplen - nk, one(FT), lnz[ilpnt], jlen)

                        iupnt = xunz[kfirst]
                        if  (klen > nups)
                            dgemm("n", "t", klen - nups, nups, nk, - one, unz[kupnt + nups], ksuplen - nk, lnz(klpnt), ksuplen, one, unz(iupnt), jlen - nj)
                        end
                    else
# -----------------------------------
#                      general sparse cmod(jsup, ksup).
#                      compute cmod(jsup, ksup) update
#                      in work storage.
# -----------------------------------
                        store = klen * nups
                        if  (store > tmpsiz)
                        iflag = - 2
                        end

# ---------------------------------
#                      gather indices of ksup relative
#                      to jsup.
# ---------------------------------
                        igathr(klen, lindx(kxpnt), map, relind)

                        dgemm("n", "t", klen, nups, nk, -one(FT), lnz[klpnt], ksuplen, unz[kupnt], ksuplen - nk, zero(FT), temp, klen)

# -----------------------------------------
#                      incorporate the cmod(jsup, ksup) block
#                      update into the appropriate columns of l.
# -----------------------------------------
                        assmb(klen, nups, temp, view(relind, 1:end), view(relind, 1:end), view(xlnz, fj:end), lnz, jlen)

                        if  (klen > nups)

                            dgemm ("n", "t", klen - nups, nups, nk, -one(FT), unz[kupnt + nups], ksuplen - nk, lnz(klpnt), ksuplen, zero(FT), temp, klen - nups)

# --
#                         incorporate the cmod(jsup, ksup) block
#                         update into the appropriate rows of u.
# -
                            assmb(klen - nups, nups, temp, view(relind, 1:end), view(relind, nups + 1:end), view(xunz, fj:end), unz, jlen)
                        end
                    end
                end
            end

# -
#             link ksup into linked list of the next supernode
#             it will update and decrement ksup"s active
#             lngth.
# -
            if  (klen > nups)
                nxtsup = snode[nxt]
                link[ksup] = link[nxtsup]
                link[nxtsup] = ksup
                lngth[ksup] = klen - nups
            else
                lngth[ksup] = 0
            end

# -----------------------------
#             next updating supernode (ksup).
# 
        end

# 
#          apply partial lu to the diagonal block.
# 
        dgetrf(nj, nj, lnz[jlpnt], jlen, ipvt[fj], iflag)

        if  (iflag != 0)
            iflag = - 1
        end
# ---------------
#          update columns.
# ---------------
        dtrsm("r", "u", "n", "n", jlen - nj, nj, one(FT), lnz[jlpnt], jlen, lnz[jlpnt + nj], jlen)

# --------------------------
#          apply permutation to rows.
# --------------------------
        if  (jlen > nj)
            luswap(jlen - nj, nj, unz[jupnt], jlen - nj, ipvt[fj])

# ---------------
#             update rows.
# ---------------
            dtrsm("r", "l", "t", "u", jlen - nj, nj, one, lnz[jlpnt], jlen, unz[jupnt], jlen - nj)
        end

# 
#          insert jsup into linked list of first supernode
#          it will update.
# 
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

#
#
# *     LUSolve ... block triangular solutions ^^^^^^^^^^ *
#
# """  Purpose:
#      Given the L U factorization of a sparse structurally symmetric
#      positive definite matrix, this subroutine performs the
#      triangular solution.  It uses output from LUFactor.
#  Input parameters:
#      nsuper - number of supernodes.
#      xsuper - supernode partition.
#      (xlindx, lindx) - row indices for each supernode.
#      (xlnz, lnz) - Cholesky factor.
#  Updated parameters:
#      rhs - on input, contains the right hand side.  on
#                           output, contains the solution.
# """
"""
"""
function lulsolve(nsuper, xsuper, xlindx, lindx, xlnz, lnz, ipiv, rhs)
     integer :: nsuper
     integer :: lindx(*), ipiv(*), xsuper(*), xlindx(*), xlnz(*)
     real(double) :: lnz(*), rhs(*)

     integer :: fj, isub, j, jj, jlen, jlpnt, jsup, jxpnt, nj
     integer :: length, maxlength
     real(double), dimension(:), allocatable :: temp

# ----------
#    constants.
# ----------
     real(double) :: one, zero
     parameter (one  = 1.0, zero = 0.0)

#

     if  (nsuper < = 0)  return

     length = 0; maxlength = 0
     for j = 1: nsuper
        length = xlindx(j + 1) - xlindx(j)
        maxlength = max(length, maxlength)
     end

     FIXME allocate(temp(maxlength))

# ------------------------
#    forward substitution ...
# ------------------------
     for jsup = 1: nsuper

        fj    = xsuper(jsup)
        nj    = xsuper(jsup + 1) - fj
        jlen  = xlnz(fj + 1) - xlnz(fj)
        jxpnt = xlindx(jsup)
        jlpnt = xlnz(fj)

# -
#       pivot rows of RHS to match pivoting of L and U.
# -
        dlaswp (1, rhs(fj), nj, 1, nj, ipiv(fj), 1)

        dtrsm ("left", "lower", "no transpose", "unit", nj, 1, one, lnz(jlpnt), jlen, rhs(fj), nj)

        dgemv ("no transpose", jlen - nj, nj, - one, lnz(jlpnt + nj), jlen, rhs(fj), 1, zero, temp, 1)

        jj = jxpnt + nj - 1
        for  j = 1: jlen - nj
           jj = jj + 1
           isub = lindx(jj)
           rhs(isub) = rhs(isub) + temp(j)
           temp(j) = 0.0d0
        end

     end

     deallocate(temp)

   end
"""
"""
function luusolve(n, nsuper, xsuper, xlindx, lindx, xlnz, lnz, xunz, unz, rhs)
     integer :: nsuper, n
     integer :: lindx(*), xsuper(*), xlindx(*), xlnz(*), xunz(*)
     real(double) :: lnz(*), rhs(*), unz(*)

     integer :: fj, isub, j, jj, jlen, jlpnt, jsup, jupnt, jxpnt, nj
     integer :: length, maxlength
     real(double), dimension(:), allocatable :: temp

# ----------
#    constants.
# ----------
     real(double) :: one, zero
     parameter (one  = 1.0, zero = 0.0)

#

     if  (nsuper < = 0)  return

     length = 0; maxlength = 0
     for j = 1: nsuper
        length = xlindx(j + 1) - xlindx(j)
        maxlength = max(length, maxlength)
     end

     FIXME allocate(temp(maxlength))

# -------------------------
#    backward substitution ...
# -------------------------
     for jsup = nsuper, 1: - 1

        fj    = xsuper(jsup)
        nj    = xsuper(jsup + 1) - fj
        jlen  = xlnz(fj + 1) - xlnz(fj)
        jxpnt = xlindx(jsup)
        jlpnt = xlnz(fj)
        jupnt = xunz(fj)

        jj = jxpnt + nj - 1
        for j = 1: jlen - nj
           jj = jj + 1
           isub = lindx(jj)
           if (isub > n)
              fatal("index out of bounds in rhs", isub)
           end
           temp(j) = rhs(isub)
        end  (jlen > nj)
           dgemv ("transpose", jlen - nj, nj, - one, unz(jupnt), jlen - nj, temp, 1, one, rhs(fj), 1)
        end

        dtrsm ("left", "upper", "no transpose", "non - unit", nj, 1, one, lnz(jlpnt), jlen, rhs(fj), nj)

     end

     deallocate(temp)

   end
"""
"""
function lutransposelsolve(nsuper, xsuper, xlindx, lindx, xunz, unz, xlnz, lnz, ipiv, rhs)
     integer :: nsuper
     integer :: lindx(*), ipiv(*), xsuper(*), xlindx(*), xunz(*), xlnz(*)
     real(double) :: unz(*), rhs(*), lnz(*)

     integer :: fj, isub, j, jj, jlen, jlpnt, jsup, jxpnt, nj, jupnt, juen
     integer :: length, maxlength
     real(double), dimension(:), allocatable :: temp

# ----------
#    constants.
# ----------
     real(double) :: one, zero
     parameter (one  = 1.0, zero = 0.0)

#

     if  (nsuper < = 0)  return

     length = 0; maxlength = 0
     for j = 1: nsuper
        length = xlindx(j + 1) - xlindx(j)
        maxlength = max(length, maxlength)
     end

     FIXME allocate(temp(maxlength))

# ------------------------
#    forward substitution ...
# ------------------------
     for jsup = 1: nsuper
        fj    = xsuper(jsup)
        nj    = xsuper(jsup + 1) - fj
        jlen  = xlnz(fj + 1) - xlnz(fj)
        juen  = xunz(fj + 1) - xunz(fj)
        jxpnt = xlindx(jsup)
        jlpnt = xlnz(fj)
        jupnt = xunz(fj)

        dtrsm ("left", "upper", "transpose", "non - unit", nj, 1, one, lnz(jlpnt), jlen, rhs(fj), nj)

        if (jlen > nj)
            dgemv ("no transpose", juen, nj, - one, unz(jupnt), juen, rhs(fj), 1, zero, temp, 1)
        end

        jj = jxpnt + nj - 1
        for  j = 1: juen
           jj = jj + 1
           isub = lindx(jj)
           rhs(isub) = rhs(isub) + temp(j)
           temp(j) = 0.0d0
        end

     end

     deallocate(temp)

   end
"""
"""
function lutransposeusolve(n, nsuper, xsuper, xlindx, lindx, xunz, unz, xlnz, lnz, rhs, ipiv)
     integer :: nsuper, n
     integer :: lindx(*), xsuper(*), xlindx(*), xlnz(*), xunz(*), ipiv(*)
     real(double) :: lnz(*), rhs(*), unz(*)

     integer :: fj, isub, j, jj, jlen, jlpnt, jsup, jupnt, jxpnt, nj, juen
     integer :: length, maxlength
     real(double), dimension(:), allocatable :: temp

# ----------
#    constants.
# ----------
     real(double) :: one, zero
     parameter (one  = 1.0, zero = 0.0)

#

     if  (nsuper < = 0)  return

     length = 0; maxlength = 0
     for j = 1: nsuper
        length = xlindx(j + 1) - xlindx(j)
        maxlength = max(length, maxlength)
     end

     FIXME allocate(temp(maxlength))

# -------------------------
#    backward substitution ...
# -------------------------
     for jsup = nsuper, 1: - 1
        fj    = xsuper(jsup)
        nj    = xsuper(jsup + 1) - fj
        jlen  = xlnz(fj + 1) - xlnz(fj)
        juen  = xunz(fj + 1) - xunz(fj)
        jxpnt = xlindx(jsup)
        jlpnt = xlnz(fj)
        jupnt = xunz(fj)

        jj = jxpnt + nj - 1
        for j = 1: jlen - nj
           jj = jj + 1
           isub = lindx(jj)

           if (isub > n)
              fatal("index out of bounds in rhs", isub)
           end
           temp(j) = rhs(isub)

        end  (jlen > nj)
            dgemv ("transpose", jlen - nj, nj, - one, lnz(jlpnt + nj), jlen, temp, 1, one, rhs(fj), 1)
        end

        dtrsm ("left", "lower", "transpose", "unit", nj, 1, one, lnz(jlpnt), jlen, rhs(fj), nj)

# -
#       pivot rows of RHS to match pivoting of L and U.
# -
        dlaswp (1, rhs(fj), nj, nj, 1, ipiv(fj), - 1)

     end

     deallocate(temp)

   end
#
end 



