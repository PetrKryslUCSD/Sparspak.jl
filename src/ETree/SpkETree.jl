""" 
This module contains a collection of subroutines for finding and manipulating
elimination trees, finding (weighted) postorderings of them, and related
functions.
"""

module SpkETree

using ..SpkOrdering: Ordering
using ..SpkGraph: Graph

mutable struct ETree{IT}
    nv::IT
    parent::Vector{IT}
end

"""
    ETree(nv::IT) where {IT}

Construct elimination tree.
"""
function ETree(nv::IT) where {IT}
    parent = fill(zero(IT), nv)
    return ETree(nv, parent)
end

"""
  Given a graph and an ordering, GetETree finds the corresponding
  elimination tree. It calls the subroutine FindETree, which actually
  does the work.

Input Parameter:
  g - the graph whose elimination tree is to be found.
  order - the ordering for g

Updated Parameters:
   t - the elimination tree.
"""
function getetree(g::Graph, order::Ordering, t::ETree)
    findetree(g.nv, g.xadj, g.adj, order.rperm, order.rinvp, t.parent)
end

"""
  To determine the elimination tree from a given ordering and the
  adjacency structure. The parent vector is returned.

Input Parameters:
  n - number of equations.
  (xadj, adj) - the adjacency structure.
  (rPerm, rInvp) - permutation and inverse permutation vectors

Output Parameters:
  parent - the parent vector of the elimination tree.

Working Storage:
   ancestor - the ancestor vector.
"""
function findetree(n, xadj, adj, rperm, rinvp, parent)
#
    ancestor = fill(zero(eltype(xadj)), n)
    for i in 1:n
        parent[i] = 0; ancestor[i] = 0; vertex = rperm[i]
        for j = xadj[vertex]:(xadj[vertex + 1] - 1)
            nbr = rinvp[adj[j]]
            if (nbr < i)
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#                   For each nbr, find the root of its current
#                   elimination tree.  Perform path compression
#                   as the subtree is traversed.
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                while(ancestor[nbr] != 0 && ancestor[nbr] != i)
                    next = ancestor[nbr]; ancestor[nbr] = i; nbr = next
                end
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#                   Now nbr is the root of the subtree.  Make i
#                   the parent vertex of this root.
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                if (ancestor[nbr] == 0)
                    parent[nbr] = i; ancestor[nbr] = i
                end
            end
        end
    end
end

"""
  GetPostorder finds a postordering of elimination tree t. The resulting
  ordering is returned in the object ``order"". The vector argument
  ``weight"" is optional. If it is present,  the postordering will
  be one where the child vertices are ordered in increasing order of
  their weights.
  The elimination tree is reordered according to the postordering.

Input Parameters:
  t - the e - tree for which the postordering is found.
  weight - an optional weighting on the tree

Updated Parameters:
  order - the required ordering
"""
function getpostorder(t::ETree{IT}, order::Ordering, weight) where {IT}
    firstson = fill(zero(IT), t.nv)
    brother = fill(zero(IT), t.nv)
    stack = fill(zero(IT), t.nv)
    invpos = fill(zero(IT), t.nv)

    weightedbinarytree(t.nv, t.parent, weight, firstson, brother)

    postordertree(t.nv, firstson, brother, invpos, t.parent, stack)

    stack[invpos[1:t.nv]] = weight[1:t.nv]
    weight[1:t.nv] = stack[1:t.nv]

    stack[1:t.nv] = invpos[order.rinvp[1:t.nv]]
    order.rinvp[1:t.nv] = stack[1:t.nv]
    order.rperm[order.rinvp[1:t.nv]] = 1:t.nv
    order.cperm = order.rperm; order.cinvp = order.rinvp
end

function getpostorder(t::ETree{IT}, order::Ordering) where {IT}
    firstson = fill(zero(IT), t.nv)
    brother = fill(zero(IT), t.nv)
    stack = fill(zero(IT), t.nv)
    invpos = fill(zero(IT), t.nv)

    binarytree(t.nv, t.parent, firstson, brother)

    postordertree(t.nv, firstson, brother, invpos, t.parent, stack)

    stack[1:t.nv] = invpos[order.rinvp[1:t.nv]]
    order.rinvp[1:t.nv] = stack[1:t.nv]
    order.rperm[order.rinvp[1:t.nv]] = 1:t.nv
    order.cperm = order.rperm; order.cinvp = order.rinvp
end

"""
  To determine the binary tree representation of the elimination
  tree given by the parent vector.  The returned representation
  will be given by the first - son and brother vectors.  The root
  of the binary tree is always n.

Input Parameters:
  n - number of equations.
  parent - the parent vector of the elimination tree.
            It is assumed that parent(i) > i except for the roots.

Output Parameters:
  fson - the first son vector.
   brother - the brother vector.
"""
function binarytree(n, parent, fson, brother)
    fson .= zero(eltype(parent)); brother .= zero(eltype(parent)); lroot = n
# --
#       for each vertex : = n - 1 step - 1 downto 1, do the following.
# --
    for vertex = (n - 1):-1:1
        vpar = parent[vertex]
        if (vpar <= 0 || vpar == vertex)
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#               vertex has no parent.  Given structure is a forest.
#               Set vertex to be one of the roots of the trees.
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            brother[lroot] = vertex; lroot = vertex
        else
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#               otherwise, becomes first son of its parent.
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            brother[vertex] = fson[vpar]; fson[vpar] = vertex
        end
    end
    brother[lroot] = 0
end

"""
  Based on the binary representation (first - son, brother) of the
  elimination tree, a postordering is determined. The corresponding
  parent vector is also modified to reflect the reordering.

Input Parameters:
  root - root of the elimination tree (usually n).
  fson - the first son vector.
  brother - the brother vector.

Updated Parameters:
  parent - the parent vector.

Output Parameters:
  rInvpos - inverse permutation for the postordering.

Working Parameters:
   stack - the stack for postorder traversal of the tree.
"""
function postordertree(root, fson, brother, rinvpos, parent, stack)
    num = 0; top = 0; vertex = root
# --
#       traverse along the first sons pointer and push the tree vertices
#       along the traversal into the stack.
# -  --
    while(vertex > 0)

        while(vertex > 0)
            top = top + 1; stack[top] = vertex; vertex = fson[vertex]
        end

        while(vertex <= 0 && top > 0)
            vertex = stack[top]; top = top - 1
            num = num + 1; rinvpos[vertex] = num
#
#               then, traverse to its younger brother if it has one.
#
            vertex = brother[vertex]
        end

    end

    for vertex in 1:num
        nuvertex = rinvpos[vertex]; vpar = parent[vertex]
        if (vpar > 0) vpar = rinvpos[vpar]; end
        brother[nuvertex] = vpar
    end

    parent .= brother
end


"""
  To determine a binary tree representation of the elimination
  tree, for which every "last child" has the maximum possible
  column nonzero count in the factor.  The returned representation
  will be given by the first - son and brother vectors.  The root of
  the binary tree is always n.

Input Parameters:
  n - number of equations.
  parent - the parent vector of the elimination tree.
            It is assumed that parent(i) > i except for the roots.
  weight - a weighting on the tree.

Output Parameters:
  fson - the first son vector.
  brother - the brother vector.

Working Storage:
   lson - last son vector.
"""
function weightedbinarytree(n, parent, weight, fson, brother)
#
    lson = fill(zero(eltype(fson)), n)
    fson .= zero(eltype(fson)); brother .= zero(eltype(fson)); 
    lson .= zero(eltype(fson)); lroot = n
# --
#       for each vertex : = n - 1 step - 1 downto 1, do the following.
# --
    for vertex in (n - 1):-1:1
        vpar = parent[vertex]
        if (vpar <= 0 || vpar == vertex)
#
#               vertex has no parent.  Given structure is a forest.
#               set vertex to be one of the roots of the trees.
# 
            brother[lroot] = vertex; lroot = vertex
        else
# 
#               otherwise, becomes first son of its parent.
# 
            vlastson = lson[vpar]
            if (vlastson != 0)
                if (weight[vertex] >= weight[vlastson])
                    brother[vertex] = fson[vpar]; fson[vpar] = vertex
                else
                    brother[vlastson] = vertex; lson[vpar] = vertex
                end
            else
                fson[vpar] = vertex; lson[vpar] = vertex
            end
        end
    end

    brother[lroot] = 0

end

end