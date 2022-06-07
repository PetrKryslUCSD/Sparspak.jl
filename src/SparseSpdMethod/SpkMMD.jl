# A collection of routines to find an MMD (multiple minimum degree) ordering.
# There are no declarations here.

# Multiple minimum degree Ordering algorithm (MMD).
# Written by Erik Demaine, eddemaine@uwaterloo.ca
# Based on a Fortran 77 code written by Joseph Liu.
# For information on the minimum degree algorithm see the articles:
# The evolution of the minimum degree algorithm by Alan George and
# Joseph Liu, SIAM Rev. 31 pp. 1 - 19, 1989.0
# Modification of the minimum degree algorithm by multiple
# elimination, ACM Trans. Math. Soft. 2 pp.141 - 152, 1985

module SpkMmd

using OffsetArrays
using ..SpkOrdering: Ordering
using ..SpkGraph: Graph

function mmd(g::Graph, order::Ordering)
    generalmmd(g.nv, g.xadj, g.adj, order.rperm, order.rinvp)
    order.cinvp .= order.rinvp
    order.cperm .= order.rperm
end

"""     
    generalmmd(n, xadj, adj, perm, invp)

This routine implements the minimum degree algorithm.  It makes use of the
implicit representation of elimination graphs by quotient graphs, and the
notion of indistinguishable nodes.  It also implements the modifications by
multiple elimination and minimum external degree.

Input parameters -
- n - number of equations
  (xadj, adj) - adjacency structure for the graph.
  delta - tolerance value for multiple elimination. FIX ME: should delta be passed as argument?

Output:
  perm, invp - the minimum degree Ordering.

Working arrays -
  degHead (deg) - points to first node with degree deg, or 0 if there
               are no such nodes.
  degNext (node) - points to the next node in the degree list
               associated with node, or 0 if node was the last in the
               degree list.
  degPrev (node) - points to the previous node in a degree list
               associated with node, or the negative of the degree of
               node (if node was the last in the degree list), or 0
               if the node is not in the degree lists.
  superSIze - the size of the supernodes.
  elimHead - points to the first node eliminated in the current pass
               Using elimNext, one can determine all nodes just
               eliminated.
  elimNext (node) - points to the next node in a eliminated supernode
               or 0 if there are no more after node.
  marker - marker vector.
  mergeParent - the parent map for the merged forest.
    needsUpdate (node) - > 0 iff node needs degree update.(0 otherwise)
"""
function generalmmd(n::IT, xadj::Vector{IT}, adj::Vector{IT}, perm::Vector{IT}, invp::Vector{IT}) where {IT}
    delta = zero(eltype(xadj))
    maxint = typemax(eltype(xadj))
    #
    #       Copy adjacency structure so that we can modify it.
    #
    adjncy = deepcopy(adj[1:(xadj[n+1]-1)])

    deghead1 = fill(zero(IT), n)
    deghead = OffsetArray(deghead1, 0:(n-1))

    marker = fill(zero(IT), n)
    mergeparent = fill(zero(IT), n)
    needsupdate = fill(zero(IT), n)
    degnext = fill(zero(IT), n)
    degprev = fill(zero(IT), n)
    supersize = fill(zero(IT), n)
    elimnext = fill(zero(IT), n)
    #
    #       Initialization for the minimum degree algorithm.
    #
    deghead .= 0
    supersize .= 1
    marker .= 0
    elimnext .= 0
    mergeparent .= 0         # Initially there is no merging
    needsupdate .= 0         # Initially no nodes need a degree update
    #
    #       Initialize the degree doubly linked lists.
    #
    for node in 1:n
        ndeg = xadj[node+1] - xadj[node]; fnode = deghead[ndeg]
        deghead[ndeg] = node; degnext[node] = fnode
        if (fnode > 0)
            degprev[fnode] = node
        end
        degprev[node] = -ndeg
    end
    #
    #       num counts the number of ordered nodes plus 1.0
    #
    num = 1
    #
    #       Eliminate all isolated nodes.
    #
    invp[1:n] .= 0
    mdnode = deghead[0]
    while (mdnode > 0)
        marker[mdnode] = maxint
        invp[mdnode] = num
        num = num + 1
        mdnode = degnext[mdnode]
    end
    deghead[0] = 0
    #
    #       Search for node of the minimum degree.
    #       mindeg is the current minimum degree;
    #       tag is used to facilitate marking nodes.
    #
    tag = 1
    mindeg = 1   # We"ve already eliminated all 0 - degree (isolated) nodes
    while (num <= n)
        while (deghead[mindeg] <= 0)
            mindeg = mindeg + 1
        end
        #  
        # Use value of delta to set up mindegLimit, which governs when a degree
        # update is to be performed.
        #  
        mindeglimit = mindeg + delta
        if (delta < 0)
            mindeglimit = mindeg
        end
        elimhead = 0

        while true
            #  -
            # Find a node of minimum degree, say mdNode.
            #  -
            mdnode = deghead[mindeg]
            while (mdnode <= 0)
                mindeg = mindeg + 1
                if (mindeg > mindeglimit)
                    @goto pass
                end
                mdnode = deghead[mindeg]
            end
            #  -
            # Remove mdNode from the degree structure.
            #  -
            mdnextnode = degnext[mdnode]
            deghead[mindeg] = mdnextnode
            if (mdnextnode > 0)
                degprev[mdnextnode] = -mindeg
            end
            invp[mdnode] = num
            if (num + supersize[mdnode] > n)
                @goto main
            end
            #
            # Eliminate mdNode and perform quotient Graph transformation.  Reset
            # tag value if necessary.
            #
            tag = tag + 1
            if (tag >= maxint)
                tag = 1
                marker[findall(marker .< maxint)] .= 0
            end
            
            mmdelim(mdnode, xadj, adjncy, deghead, degnext, degprev, supersize, elimnext, marker, tag, mergeparent, needsupdate, invp)
            
            num = num + supersize[mdnode]
            #  -
            # Add mdNode to the list of nodes eliminated in this pass.
            #  -
            elimnext[mdnode] = elimhead
            elimhead = mdnode
        end
        @label pass
        #  -
        #           Update degrees of the nodes involved in the
        #           minimum - degree nodes elimination.
        #  -
        if (num > n)
            @goto main
        end
        mindeg, tag = mmdupdate(elimhead, n, xadj, adjncy, delta, mindeg, deghead, degnext, degprev, supersize, elimnext, marker, tag, mergeparent, needsupdate, invp)
    end # 
    @label main
    mmdnumber(n, perm, invp, mergeparent)
    return true
end

"""    Purpose - This routine eliminates the node mdNode of
      minimum degree from the adjacency structure, which
      is stored in the quotient Graph format.  It also
      transforms the quotient Graph representation of the
      elimination Graph.
   Input parameters -
      mdNode - node of minimum degree.
      tag - tag value.
      invp - the inverse of an incomplete minimum degree Ordering.
                (It is zero at positions where the Ordering is unknown.)
   Updated parameters -
      (xadj, adjncy) - updated adjacency structure (xadj is not updated).
      degHead (deg) - points to first node with degree deg, or 0 if there
                   are no such nodes.
      degNext (node) - points to the next node in the degree list
                   associated with node, or 0 if node was the last in the
                   degree list.
      degPrev (node) - points to the previous node in a degree list
                   associated with node, or the negative of the degree of
                   node (if node was the last in the degree list), or 0
                   if the node is not in the degree lists.
      superSIze - the size of the supernodes.
      elimNext (node) - points to the next node in a eliminated supernode
                   or 0 if there are no more after node.
      marker - marker vector.
      mergeParent - the parent map for the merged forest.
       needsUpdate (node) - > 0 iff node needs update. (0 otherwise)
"""
function mmdelim(mdnode, xadj, adjncy, deghead, degnext, degprev, supersize, elimnext, marker, tag, mergeparent, needsupdate, invp)
    maxint = typemax(eltype(xadj))
# 
#       Find reachable set and place in data structure.
# 
    marker[mdnode] = tag
# 
#       elmnt points to the beginning of the list of eliminated
#       neighbors of mdNode, and rloc gives the storage
#       location for the next reachable node.
# 
    elmnt = 0
    rloc = xadj[mdnode] ; rlmt = xadj[mdnode + 1] - 1
    for i in xadj[mdnode]:(xadj[mdnode + 1] - 1)
        neighbor = adjncy[i]
        if (neighbor == 0) break; end
        if (marker[neighbor] < tag)
            marker[neighbor] = tag
            if (invp[neighbor] == 0)
                adjncy[rloc] = neighbor ; rloc = rloc + 1
            else
                elimnext[neighbor] = elmnt ; elmnt = neighbor
            end
        end
    end
# 
#       Merge with reachable nodes from generalized elements.
# 
    while (elmnt > 0)
        adjncy[rlmt] = - elmnt
        j = xadj[elmnt] ; jstop = xadj[elmnt + 1] ; node = adjncy[j]
        while (node != 0)
            if (node < 0)
                j = xadj[- node] ; jstop = xadj[- node + 1]
            else
                if (marker[node] < tag && degnext[node] >= 0)
                    marker[node] = tag
# 
#                       Use storage from eliminated nodes if necessary.
# 
                    while (rloc >= rlmt)
                        link = - adjncy[rlmt]
                        rloc = xadj[link] ; rlmt = xadj[link + 1] - 1
                    end
                    adjncy[rloc] = node ; rloc = rloc + 1
                end
                j = j + 1
            end
            if (j >= jstop) break; end
            node = adjncy[j]
        end
        elmnt = elimnext[elmnt]
    end 
    if (rloc <= rlmt) adjncy[rloc] = 0; end
# 
#       For each node in the reachable set, do the following ...
# 
    i = xadj[mdnode] ; istop = xadj[mdnode + 1] ; rnode = adjncy[i]
    while (rnode != 0)
        if (rnode < 0)
            i = xadj[- rnode] ; istop = xadj[- rnode + 1]
        else
# 
#               If rnode is in the degree list structure ...
# 
            pvnode = degprev[rnode]
            if (pvnode != 0)
# 
#                   Then remove rnode from the structure.
# 
                nxnode = degnext[rnode]
                if (nxnode > 0) degprev[nxnode] = pvnode; end
                if (pvnode > 0)
                    degnext[pvnode] = nxnode
                else
                    deghead[- pvnode] = nxnode
                end
            end
# 
#               Purge inactive quotient neighbors of rnode.
# 
            xqnbr = xadj[rnode]
            for j in xadj[rnode]:(xadj[rnode + 1] - 1)
                neighbor = adjncy[j]
                if (neighbor == 0) break; end
                if (marker[neighbor] < tag)
                    adjncy[xqnbr] = neighbor ; xqnbr = xqnbr + 1
                end
            end
# 
#               If no active neighbor after the purging ...
# 
            nqnbrs = xqnbr - xadj[rnode]
            if (nqnbrs <= 0)
# 
#                   Then merge rnode with mdNode.
# 
                supersize[mdnode] = supersize[mdnode] + supersize[rnode]
                supersize[rnode] = 0 ; mergeparent[rnode] = mdnode
                marker[rnode] = maxint
            else
# 
#                   Else flag rnode for degree update, and
#                   add mdNode as a neighbor of rnode.
# 
                needsupdate[rnode] = nqnbrs + 1
                adjncy[xqnbr] = mdnode ; xqnbr = xqnbr + 1
                if (xqnbr < xadj[rnode + 1]) adjncy[xqnbr] = 0; end
            end
            degprev[rnode] = 0 ; i = i + 1
        end
        if (i >= istop) break; end
        rnode = adjncy[i]
    end
end

"""    Purpose - This routine updates the degrees of nodes
      after a multiple elimination step.
   Input parameters -
      elimHead - the beginning of the list of eliminated
               nodes (i.e., newly formed elements).
      neqns - number of equations.
      (xadj, adjncy) - adjacency structure.
      delta - tolerance value for multiple elimination.
      invp - the inverse of an incomplete minimum degree Ordering.
               (It is zero at positions where the Ordering is unknown.)
   Updated parameters -
      mindeg - new minimum degree after degree update.
      degHead (deg) - points to first node with degree deg, or 0 if there
                   are no such nodes.
      degNext (node) - points to the next node in the degree list
                   associated with node, or 0 if node was the last in the
                   degree list.
      degPrev (node) - points to the previous node in a degree list
                   associated with node, or the negative of the degree of
                   node (if node was the last in the degree list), or 0
                   if the node is not in the degree lists.
      superSIze - the size of the supernodes.
      elimNext (node) - points to the next node in a eliminated supernode
                   or 0 if there are no more after node.
      marker - marker vector for degree update.
      tag - tag value.
      mergeParent - the parent map for the merged forest.
       needsUpdate (node) - > 0 iff node needs update. (0 otherwise)
"""
function mmdupdate(elimhead, neqns, xadj, adjncy, delta, mindeg, deghead, degnext, degprev, supersize, elimnext, marker, tag, mergeparent, needsupdate, invp)
    maxint = typemax(eltype(xadj))
#
    mindeglimit = mindeg + delta

    deg = zero(eltype(xadj))
    enode = zero(eltype(xadj))

# -
#       For each of the newly formed element, do the following.
# -
    elimnode = elimhead
    while (elimnode > 0)
# -
#           (Reset tag value if necessary.)
# -
        mtag = tag + mindeglimit
        if (mtag >= maxint)
            tag = 1 ; mtag = tag + mindeglimit
            marker[findall(marker .< maxint)] .= 0
        end
# -
#           Create two linked lists from nodes associated with elimNode: one
#           with two neighbors (q2head) in adjacency structure, and the other
#           with more than two neighbors (qxhead).  Also compute elimsize,
#           the number of nodes in this element.
# -
        q2head = 0 ; qxhead = 0 ; elimsize = 0
        i = xadj[elimnode] ; istop = xadj[elimnode + 1] ; enode = adjncy[i]
        while (enode != 0)
            if (enode < 0)
                i = xadj[- enode] ; istop = xadj[- enode + 1]
            else
                if (supersize[enode] != 0)
                    elimsize = elimsize + supersize[enode] ;
                    marker[enode] = mtag
# -
#                       If enode requires a degree update,
#                        do the following.
# -
                    if (needsupdate[enode] > 0)
# -
#                           Place either in qxhead or q2head lists.
# -
                        if (needsupdate[enode] != 2)
                            elimnext[enode] = qxhead ; qxhead = enode
                        else
                            elimnext[enode] = q2head ; q2head = enode
                        end
                    end
                end
                i = i + 1
            end
            if (i >= istop) break; end
            enode = adjncy[i]
        end
# -
#           For each enode in q2 list, do the following.
# -
        enode = q2head
        while (enode > 0)
            if (needsupdate[enode] > 0)
                tag = tag + 1 ; deg = elimsize
# -
#                   Identify the other adjacent element neighbor.
# -
                istart = xadj[enode] ; neighbor = adjncy[istart]
                if (neighbor == elimnode) neighbor = adjncy[istart + 1]; end
# -
#                   If neighbor is uneliminated, increase degree count.
# -
                if (invp[neighbor] == 0)
                    deg = deg + supersize[neighbor]
                else
# - -
#                       Otherwise, for each node in the 2nd element,
#                       do the following.
# -
                    i = xadj[neighbor] ; istop = xadj[neighbor + 1] ;
                    node = adjncy[i]
                    while (node != 0)
                        if (node < 0)
                            i = xadj[- node] ; istop = xadj[- node + 1]
                        else
                            if (node != enode && supersize[node] != 0)
# 
#                                   Case when node is not yet considered.
# 
                                if (marker[node] < tag)
                                    marker[node] = tag ;
                                    deg = deg + supersize[node]
# -
#                                       Case when node is indistinguishable
#                                       Merge them into new supernode.
# -
                                elseif (needsupdate[node] > 0)
                                    if (needsupdate[node] == 2)
# -
#                                           Case when node is not outmatched
# -
                                        supersize[enode] = supersize[enode] + supersize[node]
                                        supersize[node] = 0
                                        marker[node] = maxint
                                        mergeparent[node] = enode
                                    end
                                    needsupdate[node] = 0
                                    degprev[node] = 0
                                end
                            end
                            i = i + 1
                        end
                        if (i >= istop) break ; end
                        node = adjncy[i]
                    end
                end
                deg, mindeg = __updateexternaldegree(deg, mindeg, enode, supersize, deghead, degnext, degprev, needsupdate)
            end # if
            enode = elimnext[enode]
        end # while
# -
#           For each enode in the qx list, do the following.
# -
        enode = qxhead
        while (enode > 0)
            if (needsupdate[enode] > 0)
                tag = tag + 1 ; deg = elimsize
# -
#                   For each unmarked neighbor of enode, do the following.
# -
                for i in xadj[enode]:(xadj[enode + 1] - 1)
                    neighbor = adjncy[i]
                    if (neighbor == 0) break; end
                    if (marker[neighbor] < tag)
                        marker[neighbor] = tag
# - -
#                           If uneliminated, include it in deg count.
# - -
                        if (invp[neighbor] == 0)
                            deg = deg + supersize[neighbor]
                        else
# -
#                               If eliminated, include unmarked nodes
#                               in this element into the degree count.
# -
                            j = xadj[neighbor] ; jstop = xadj[neighbor + 1]
                            node = adjncy[j]
                            while (node != 0)
                                if (node < 0)
                                    j = xadj[- node]; jstop = xadj[- node + 1]
                                else
                                    if (marker[node] < tag)
                                        marker[node] = tag ;
                                        deg = deg + supersize[node]
                                    end
                                    j = j + 1
                                end
                                if (j >= jstop) break; end
                                node = adjncy[j]
                            end
                        end
                    end
                end
                deg, mindeg = __updateexternaldegree(deg, mindeg, enode, supersize, deghead, degnext, degprev, needsupdate)
            end
# -
#               Get next enode in current element.
# -
            enode = elimnext[enode]
        end
# -
#           Get next element in the list.
# -
        tag = mtag ; elimnode = elimnext[elimnode]
    end
    return mindeg, tag
end


function __updateexternaldegree(deg, mindeg, enode, supersize, deghead, degnext, degprev, needsupdate)
    deg = deg - supersize[enode] ; firstnode = deghead[deg]
    deghead[deg] = enode ; degnext[enode] = firstnode
    degprev[enode] = - deg ; needsupdate[enode] = 0
    if (firstnode > 0) degprev[firstnode] = enode; end
    if (deg < mindeg) mindeg = deg; end
    return deg, mindeg
end

#       Purpose - This routine performs the final step in
#       producing the permutation and inverse permutation
#       vectors in the multiple elimination version of the
#       minimum degree Ordering algorithm.
#       No parameters are in fact used, since this is a routine within
#       genmmd, but we list any modifications.
#       Input parameters -
#       neqns - number of equations.
#       Output parameters -
#       perm - the minimum degree Ordering.
#       Updated parameters -
#       invp - On input, new number for roots in merged forest.
#       On output, this plus remaining inverse of perm.
#       mergeParent - the parent map for the merged forest (compressed).
#       Working arrays -
#       mergeLastnum (r) - last number used for a merged tree rooted at r.
#       
function mmdnumber(neqns, perm, invp, mergeparent)
    
# ----------------------------------------
#           Initially, no nodes except roots in the
#           merged forest have been numbered.
# ----------------------------------------
    mergelastnum = fill(zero(eltype(perm)), neqns)
    ix = findall(mergeparent .== 0)
    mergelastnum[ix] = invp[ix]
# -------------------------------------------------------
#           For each node which has been merged, do the following.
# -------------------------------------------------------
    for node in 1:neqns
        parent = mergeparent[node]
        if (parent > 0)
# ------------------------------------------
#                   Trace the merged tree until one which has
#                   not been merged is found, it root.
# ------------------------------------------
            root = 0
            while (parent > 0)
                root = parent ; parent = mergeparent[parent]
            end
            @assert root > 0
# -----------------------------------------
#                   Number node after those already numbered
#                   in merged subtree rooted at root.
# -----------------------------------------
            num = mergelastnum[root] + 1
            invp[node] = num ; mergelastnum[root] = num
# ------------------------------------
#                   Compress the path just traversed so
#                   that future queries are faster.
# ------------------------------------
            node2 = node
            while (node2 != root)
                parent = mergeparent[node2]
                mergeparent[node2] = root ; node2 = parent
            end
        end
    end
# ------------------------------
#           Inverse invp to compute perm.
# ------------------------------
    perm[invp[1:neqns]] = 1:neqns
end


end