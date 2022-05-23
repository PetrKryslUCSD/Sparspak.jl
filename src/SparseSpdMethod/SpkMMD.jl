"""  A collection of routines to find an MMD (multiple minimum degree)
  ordering. There are no declarations here.

  Multiple minimum degree Ordering algorithm (MMD).
Written by Erik Demaine, eddemaine@uwaterloo.ca
Based on a Fortran 77 code written by Joseph Liu.
For information on the minimum degree algorithm see the articles:
The evolution of the minimum degree algorithm by Alan George and
Joseph Liu, SIAM Rev. 31 pp. 1 - 19, 1989.0
Modification of the minimum degree algorithm by multiple
elimination, ACM Trans. Math. Soft. 2 pp.141 - 152, 1985
"""
module SpkMmd

using OffsetArrays
using ..SpkOrdering: Ordering
using ..SpkGraph: Graph

function mmd(g::Graph, order::Ordering)
    generalmmd(g.nv, g.xadj, g.adj, order.rperm, order.rinvp)
    order.cinvp .= order.rinvp
    order.cperm .= order.rperm
end

"""     Purpose - This routine implements the minimum degree
      algorithm.  It makes use of the implicit representation
      of elimination graphs by quotient graphs, and the
      notion of indistinguishable nodes.  It also implements
      the modifications by multiple elimination and minimum
      external degree.
   Input parameters -
      n - number of equations
      (xadj, adj) - adjacency structure for the graph.
      delta - tolerance value for multiple elimination.
   Output parameters -
      order - the minimum degree Ordering.
   Subroutines called -
      MMDElim, MMDNumber, MMDUpdate.
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
function generalmmd(n, xadj, adj, perm, invp)
    delta = zero(eltype(xadj))
    maxint = typemax(eltype(xadj))
    #
    #       Copy adjacency structure so that we can modify it.
    #
    IT = eltype(adj)
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
        ndeg = xadj[node+1] - xadj[node]
        fnode = deghead[ndeg]
        deghead[ndeg] = node
        degnext[node] = fnode
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
        #           Use value of delta to set up mindegLimit, which governs
        #           when a degree update is to be performed.
        #  
        mindeglimit = mindeg + delta
        if (delta < 0)
            mindeglimit = mindeg
        end
        elimhead = 0

        while true
            #  -
            #               Find a node of minimum degree, say mdNode.
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
            #               Remove mdNode from the degree structure.
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
            #               Eliminate mdNode and perform quotient Graph
            #               transformation.  Reset tag value if necessary.
            #
            tag = tag + 1
            if (tag >= maxint)
                tag = 1
                marker[findall(marker .< maxint)] .= 0
            end

            mmdelim(mdnode, xadj, adjncy, deghead, degnext, degprev, supersize, elimnext, marker, tag, mergeparent, needsupdate, invp)

            num = num + supersize[mdnode]
            #  -
            #               Add mdNode to the list of nodes
            #               eliminated in this pass.
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

        mmdupdate(elimhead, n, xadj, adjncy, delta, mindeg, deghead, degnext, degprev, supersize, elimnext, marker, tag, mergeparent, needsupdate, invp)
    end # 
    @label main

end

end