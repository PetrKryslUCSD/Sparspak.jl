#
# """ This module contains a collection of subroutines for performing symbolic
#  factorization and related functions.
# """
#
module SpkSymfct

using OffsetArrays

#     This subroutine determines the column counts in
#     the Cholesky factor.  It uses an algorithm due to Joseph Liu
#     found in SIMAX 11, 1990, pages 144 - 145.0

# Input parameters:
#     (i) n - number of equations.
#     (i) xadj - array of length n + 1, containing pointers
#                         to the adjacency structure.
#     (i) adj - array of length xadj(n + 1) - 1, containing
#                         the adjacency structure.
#     (i) perm - array of length n, containing the
#                         postordering.
#     (i) invp - array of length n, containing the
#                         inverse of the postordering.
#     (i) parent - array of length n, containing the
#                         elimination tree of the postordered matrix.

# Output parameters:
#     (i) colcnt - array of length n, containing the number
#                         of nonzeros in each column of the factor,
#                         including the diagonal entry.
#     (i) nlnz - number of nonzeros in the factor, including
#                         the diagonal entries.

# Work parameters:
#     (i) marker - array of length n used to mark the
#                          vertices visited in each row subtree.
function _findcolumncounts!(n, xadj, adj, perm, invp, parent, colcnt, nlnz)
#
    marker = fill(zero(eltype(xadj)), n)
    nlnz = n
#-  -  -
#       for each row subtree ...
#-  -  -
    for i in 1:n
        marker[i] = i;   colcnt[i] = 1
#-  -  -  -  -  -  -  -  -  -  -  -  -  -
#           for each nonzero in the lower triangle ...
#-  -  -  -  -  -  -  -  -  -  -  -  -  -
        for k = xadj[perm[i]]:(xadj[perm[i] + 1] - 1)
            j = invp[adj[k]]
            if (j < i)
                while (marker[j] != i)
                    colcnt[j] = colcnt[j] + 1;   nlnz = nlnz + 1
                    marker[j] = i;   j = parent[j]
                end
            end
        end
    end
end


#     This routine performs supernodal symbolic
#     factorization on a reordered linear system.
#     This is essentially a Fortran 90 translation of a code written
#     by Esmond Ng and Barry Peyton.

# Input parameters:
#     (i) n - number of equations
#     (i) xadj - array of length n + 1 containing pointers
#                         to the adjacency structure.
#     (i) adj - array of length xadj(n + 1) - 1 containing
#                         the adjacency structure.
#     (i) perm - array of length n containing the
#                         postordering.
#     (i) invp - array of length n containing the
#                         inverse of the postordering.
#     (i) colcnt - array of length n, containing the number
#                         of nonzeros (non - empty rows) in each
#                         column of the factor,
#                         including the diagonal entry.
#     (i) nsuper - number of supernodes.
#     (i) xsuper - array of length nsuper + 1, containing the
#                         first column of each supernode.
#     (i) snode - array of length n for recording
#                         supernode membership.
#     (i) nofsub - number of subscripts to be stored in
#                         lindx.

# Output parameters:
#     (i) xlindx - array of length n + 1, containing pointers
#                         into the subscript vector.
#     (i) lindx - array of length maxsub, containing the
#                         compressed subscripts.
#     (i) xlnz - column pointers for l.

# Working parameters:
#     (i) mrglnk - array of length nsuper, containing the
#                         children of each supernode as a linked list.
#     (i) rchlnk - array of length n + 1, containing the
#                         current linked list of merged indices (the
#                         "reach" set).
#     (i) marker - array of length n used to mark indices
#                         as they are introduced into each supernode"s
#                         index set.
function _symbolicfact!(n, xadj, adj, perm, invp, colcnt, nsuper, xsuper, snode, nofsub, xlindx, lindx)
        # integer :: n, nsuper, nofsub
        # integer :: xadj(*), adj(*), perm(*), invp(*), xsuper(*)
        # integer :: colcnt(*), snode(*), xlindx(*), lindx(*)
        # integer :: mrglnk(nsuper), marker(n), rchlnk(0:n)
        # integer :: fstcol, head, i, jnzbeg, jnzend, jptr, jsup, jwidth
        # integer :: knz, knzbeg, knzend, kptr, ksup, lngth, lstcol, newi
        # integer :: nexti, node, nzbeg, nzend, pcol, psup, point, tail, width
#
#       initializations ...
#       nzend  : points to the last used slot in lindx.
#       tail   : end of list indicator (in rchlnk(*), not mrglnk(*)).
#       mrglnk : create empty lists.
#       marker : "unmark" the indices.
    marker = fill(zero(eltype(xadj)), n)
    mrglnk = fill(zero(eltype(xadj)), nsuper)
    rchlnk1 = fill(zero(eltype(xadj)), n+1)
    rchlnk = OffsetArray(rchlnk1, 0:n)
# 
    nzend = 0;   head = 0;   tail = n + 1;   point = 1;
        # for jcol in 1:n
        #    marker(jcol) = 0
        # end

    for ksup = 1: nsuper
        mrglnk[ksup] = 0; fstcol = xsuper[ksup]
        xlindx[ksup] = point; point = point + colcnt[fstcol]
    end
    xlindx[nsuper + 1] = point
# -  
#       for each supernode ksup ...
# 

    for ksup in 1:nsuper
# 
#           initializations ...
#           fstcol : first column of supernode ksup.
#           lstcol : last column of supernode ksup.
#           knz    : will count the nonzeros of l in column kcol.
#           rchlnk : initialize empty index list for kcol.
# 
        fstcol = xsuper[ksup];   lstcol = xsuper[ksup + 1] - 1
        width = lstcol - fstcol + 1;   lngth = colcnt[fstcol]
        knz = 0;   rchlnk[head] = tail;   jsup = mrglnk[ksup]
# 
#           if ksup has children in the supernodal e - tree ...
# 
        if (jsup > 0)
#-  -  -  -  -  -  -  -  -  -  -  -
#               copy the indices of the first child jsup into the
#               linked list, and mark each with the value ksup.
#-  -  -  -  -  -  -  -  -  -  -  -
            jwidth = xsuper[jsup + 1] - xsuper[jsup]
            jnzbeg = xlindx[jsup] + jwidth; jnzend = xlindx[jsup + 1] - 1
            for jptr in jnzend:-1:jnzbeg
                newi = lindx[jptr];   knz = knz + 1
                marker[newi] = ksup;   rchlnk[newi] = rchlnk[head]
                rchlnk[head] = newi
            end
#-  -  -  -  -  -  -  -  -
#               for each subsequent child jsup of ksup ...
#-  -  -  -  -  -  -  -  -
            jsup = mrglnk[jsup]
            while (jsup != 0 && knz<lngth)
#-  -  -  -  -  -  -
#                   merge the indices of jsup into the list,
#                   and mark new indices with value ksup.
#-  -  -  -  -  -  -
                jwidth = xsuper[jsup + 1] - xsuper[jsup]
                jnzbeg = xlindx[jsup] + jwidth
                jnzend = xlindx[jsup + 1] - 1;   nexti = head
                for jptr in jnzbeg:jnzend
                    newi = lindx[jptr];   i = nexti;   nexti = rchlnk[i]
                    while(newi > nexti)
                        i = nexti;   nexti = rchlnk[i]
                    end 
                    if (newi < nexti)
                        knz = knz + 1;   rchlnk[i] = newi
                        rchlnk[newi] = nexti;   marker[newi] = ksup
                        nexti = newi
                    end
                end
                jsup = mrglnk[jsup]
            end
        end
# -  --  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#           structure of a(* , fstcol) has not been examined yet.
#           "sort" its structure into the linked list,
#           inserting only those indices not already in the list.
# -  --  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        if (knz < lngth)
            node = perm[fstcol]
            knzbeg = xadj[node];   knzend = xadj[node + 1] - 1
            for kptr in knzbeg:knzend
                newi = adj[kptr];   newi = invp[newi]
                if (newi > fstcol && marker[newi] != ksup)
# -  -  -  -  -  -  --  -
#                       position and insert newi in list
#                       and mark it with kcol.
# -  -  -  -  -  -  - -  -
                    nexti = head;   i = nexti;   nexti = rchlnk[i]
                    while (newi > nexti)
                        i = nexti;   nexti = rchlnk[i]
                    end
                    knz = knz + 1;   
                    rchlnk[i] = newi
                    rchlnk[newi] = nexti;   
                    marker[newi] = ksup
                end
            end
        end
# 
#           if ksup has no children, insert fstcol into the linked list.
# 
        if (rchlnk[head] != fstcol)
            rchlnk[fstcol] = rchlnk[head]
            rchlnk[head] = fstcol;   knz = knz + 1
        end
#-  -  -  -  -  -  -  -  -  -
#           copy indices from linked list into lindx(*).
# -  -  -  -  -  -  -  -  -  -  -
        nzbeg = nzend + 1;   nzend = nzend + knz
        if (nzend + 1 != xlindx[ksup + 1])
            error("Inconsistency in data structure.")
            return false
        end
        
        i = head 
        for kptr in nzbeg:nzend
            i = rchlnk[i];   lindx[kptr] = i 
        end
# - -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#           if ksup has a parent, insert ksup into its parent"s
#           "merge" list.
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        if (lngth > width)
            pcol = lindx[xlindx[ksup] + width];   psup = snode[pcol]
            mrglnk[ksup] = mrglnk[psup];   mrglnk[psup] = ksup
        end
    end
    return true
end

#     This routine computes the total number of supernode modifications
#     that will be performed, as well as the elements of the array
#     nmod. The array nmod is of length nsuper; nmod(i) contains the
#     number of supernode modifications to be made to supernode i.
#
# Input parameters:
#     (i) nsuper - number of supernodes.
#     (i) xsuper - array of length nsuper + 1, containing the
#                         first column of each supernode.
#     (i) snode - array of length n for recording
#                         supernode membership.
#     (i) xlindx - array of length n + 1, containing pointers
#                         into the subscript vector.
#     (i) lindx - array of length maxsub, containing the
#                         compressed subscripts.
#
# Output parameters:
#     (i) total - total number of supernode modifications to
#                         be performed.
#     (i) nmod - array containing the number of supernode
#                         modifications made to each supernode.
#
#
#         integer lindx(*), xlindx(*), snode(*), xsuper(*), nmod(*)
#         integer j, jsup, jnode, nsuper, prvsup, supsiz, total
# 
function findnumberofsupermods(nsuper, xsuper, snode, xlindx, lindx, total, nmod)
    nmod .= 0;   total = 0

    for jsup in 1:nsuper
        supsiz = xsuper[jsup + 1] - xsuper[jsup];   prvsup = 0

        for j = xlindx[jsup] + supsiz: xlindx[jsup + 1] - 1
            jnode = snode[lindx[j]]

            if  (jnode != prvsup)
                prvsup = jnode;   total = total + 1
                nmod[jnode] = nmod[jnode] + 1
            end
        end
    end
end

# this routine determines a bottom - up ordering for the
#          vertices of the supernodal elimination tree, and puts that
#          ordering in the array schedule. For information on
#          elimination trees and their representation, see the
#          comments in the module SpkETree and the literature
#          references contained there.
#          For serial computation, the best order is simply the
#          "natural order", with schedule(i) = i for i = 1, 2, ..., nsuper.
#
# Input parameters:
#     (i) nsuper - number of supernodes.
#     (i) xsuper - array of length nsuper + 1, containing the
#                         first column of each supernode.
#     (i) snode - array of length n for recording
#                         supernode membership.
#     (i) xlindx - array of length n + 1, containing pointers
#                         into the subscript vector.
#     (i) lindx - array of length maxsub, containing the
#                         compressed subscripts.
#
# Output parameters:
#     (i) schedule - array containing the order in which the
#                         supernodes are to be computed.
        # integer nsuper, lindx(*), xlindx(*), snode(*), xsuper(*), schedule(*)
        # integer parent(nsuper), count(nsuper)
        # integer k, par, current, col

function _findschedule!(nsuper, xsuper, snode, xlindx, lindx, schedule)
    count = fill(zero(IT), nsuper)
    parent = fill(zero(IT), nsuper)

    _findsupernodetree!(nsuper, xsuper, snode, xlindx, lindx, parent)

    count = 0; current = 0
    for k in 1:nsuper
        par = parent[k]
        if (par != 0) count[par] = count[par] + 1; end
        if (count[k] == 0)
            current = current + 1; schedule[current] = k
        end
    end

    for k in 1:nsuper
        col = schedule[k]; par = parent[col]
        if (par != 0)
            count[par] = count[par] - 1
            if (count[par] == 0)
                current = current + 1; schedule[current] = par
            end
        end
    end

end

# this routine determines the parent vector representation of
#          the supernodal elimination tree, and puts the result
#          in the array parent. For information on
#          elimination trees and their representation, see the
#          comments in the module SpkETree and the literature
#          references contained there.
#
# Input parameters:
#     (i) nsuper - number of supernodes.
#     (i) xsuper - array of length nsuper + 1, containing the
#                         first column of each supernode.
#     (i) snode - array of length n for recording
#                         supernode membership.
#     (i) xlindx - array of length n + 1, containing pointers
#                         into the subscript vector.
#     (i) lindx - array of length maxsub, containing the
#                         compressed subscripts.
#
# Output parameter:
#     (i) parent - array containing the the parent supernode
#                         of each supernode
#
#
#
# integer lindx(*), xlindx(*), snode(*), xsuper(*), parent(*)
# integer j, jsup, nsuper
#
function _findsupernodetree!(nsuper, xsuper, snode, xlindx, lindx, parent)
    parent[nsuper] = 0

    for jsup = 1: (nsuper - 1)
        j = xlindx[jsup] + xsuper[jsup + 1] - xsuper[jsup]
        parent[jsup] = snode[lindx[j]]
    end

end


#     This routine uses the elimination tree and the factor column
#     counts to compute the supernode partition; it assumes a
#     postordering of the elimination tree.
#     For more information on supernodes, see the article by
#     Liu, Ng and Peyton: On finding supernodes for sparse matrix
#     computations, SIMAX 14 pp. 242 - 252, 1993.0
#
#     For information on elimination trees, see the article by
#     Liu: The role of elimination trees in sparse factorization,
#     SIMAX 11 pp.134 - 172, 1990.0
#
#     In order to improve multiprocessor utilization, it is
#     helpful to split the larger supernodes into smaller ones.
#     This is done by introducing artificial supernodes.
#     The input parameter maxSize governs this process; no
#     supernode will have more than maxSize columns.
##
# Input parameters:
#     (i) n - number of equations.
#     (i) maxSize - limit on supernode size
#     (i) parent - array of length n, containing the
#                          elimination tree of the postordered matrix.
#     (i) colcnt - array of length n, containing the
#                          factor column counts: i.e., the number of
#                          nonzero entries in each column of L
#                          (including the diagonal entry).
# Output parameters:
#     (i) nofsub - number of subscripts.
#     (i) nsuper - number of supernodes
#     (i) xsuper - beginning of each supernode
#     (i) snode - array of length n for recording
#                           supernode membership.
function _findsupernodes!(n, parent, colcnt, nofsub, nsuper, xsuper, snode, maxsize)
        # integer :: n, nofsub, nsuper, kcol, jsup, k, maxsize, marker(n)
        # integer :: parent(*), colcnt(*), xsuper(*), snode(*)
        # integer :: limit, index
        # real    :: delta, supsize
    marker = fill(zero(eltype(parent)), n)
#
#       compute the fundamental supernode partition.
#-  -  -  -  -  -  -  -  -  -  -
    nsuper = 1; xsuper[1] = 1; 
    for kcol = 2:n
        if (parent[kcol - 1] == kcol && colcnt[kcol - 1] == colcnt[kcol] + 1) continue; end
        nsuper = nsuper + 1; xsuper[nsuper] = kcol
    end
    xsuper[nsuper + 1] = n + 1

# 
#       Determine where to split the supernodes using marker.
# 
    for jsup in 1:nsuper
        supsize = xsuper[jsup + 1] - xsuper[jsup]
        if (supsize > maxsize)
            delta = supsize / (1.0 + supsize / maxsize);
            limit = Int(floor(xsuper[jsup + 1] - delta / 2.0))
            k = 1; index = Int(floor(xsuper[jsup] + delta))
            while (index < limit)
                marker[index] = 1
                k = k + 1; index = Int(floor(xsuper[jsup] + k * delta))
            end
        end
    end
# 
#       Now determine the refined (final) supernode partitioning.
# 
    nsuper = 1; snode[1] = 1; nofsub = colcnt[1]
    for kcol in 2:n
        if (marker[kcol] != 1)
            if (parent[kcol - 1] == kcol)
                if (colcnt[kcol - 1] == colcnt[kcol] + 1)
                    snode[kcol] = nsuper; continue
                end
            end
        end
        nsuper = nsuper + 1; snode[kcol] = nsuper
        xsuper[nsuper] = kcol; nofsub = nofsub + colcnt[kcol]
    end
    xsuper[nsuper + 1] = n + 1

    return nofsub, nsuper
end

end 
