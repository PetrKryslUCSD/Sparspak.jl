module SpkGraph

using ..SpkUtilities: __extend
using ..SpkProblem: Problem

"""
    Graph{IT}

- `nv` - the number of vertices in the graph.
- `(xadj, adj)` - array pair storing the adjacency lists of the vertices.

The adjacency lists of the graph are stored in consecutive locations
in the array adj. The adjacency list for the `i` - th vertex in the graph
is stored in positions `adj[k]`, `k = xadj[i], .... xadj[i + 1] - 1`.

When the graph is symmetric, if vertex `i` is in vertex `j`'s adjacency
vertex `j` is in vertex `i`'s list. Using the representation above
each edge in the graph is stored twice.

There are no self - loops (no "diagonal elements") by default. If
diagonal elements are required, just input "diagonal" or "diag".

For convenience in accessing the lists, `xadj` is of length `nv + 1`, with
`xadj[nV + 1] = nEdges + 1`. Thus, accessing vertex `nv`'s list is the
same as for any other of the vertices.

Graphs are created from Problem objects, which have a certain number
of rows (`nrows`) and columns (`ncols`). These numbers are captured
and stored in Graph objects as `nrows` and `ncols` as well.
"""
mutable struct Graph{IT}
    # nv - the number of vertices in the graph.
    nv::IT
    nedges::IT
    nrows::IT
    ncols::IT
    # (xadj, adj) - array pair storing the adjacency lists of the vertices.
    xadj::Vector{IT}
    adj::Vector{IT}
end

"""
    Graph(p::Problem{IT}, diagonal=false) where {IT}

Construct a graph from a problem object.

It does not check that the problem object contains a structurally
symmetric matrix, since sometimes only the lower or upper triangle of
a symmetric matrix may be stored. There are routines in this module to
make a given graph object structurally symmetric.

Input:
- `p` - the problem object, used to create the graph
- `diagonal` - indicates that the diagonal elements are included. If
    diagonal is not given, the adjacency structure does not include
    the diagonal elements.
"""
function Graph(p::Problem{IT}, diagonal=false) where {IT}
    nv = p.ncols
    nrows = p.nrows
    ncols = p.ncols
    if (diagonal)
        nedges = p.nedges
    else
        nedges = p.nedges - p.dedges
    end
    xadj = zeros(IT, nv + 1)
    adj = zeros(IT, nedges)
    
    k = 1
    for i in 1:p.ncols
        ptr = p.head[i]
        xadj[i] = k
        while (ptr > 0)
            j = p.rowsubs[ptr]
            if (i != j || diagonal)
                adj[k] = j
                k = k + 1
            end
            ptr = p.link[ptr]
        end
    end
    
    xadj[p.ncols+1] = k
    
    return Graph(nv, nedges, nrows, ncols, xadj, adj)
end

"""
    makestructuresymmetric(g::Graph{IT}) where {IT}

Make the graph structure symmetric.
"""
function makestructuresymmetric(g::Graph{IT}) where {IT}
    if (isstructuresymmetric(g))
        return true
    end

    h = deepcopy(g)
    h.nv = g.nv
    h.nedges = g.nedges
    h.nrows = g.nrows
    h.ncols = g.ncols
    #   -  -  -
    #       The array c contains the edge counts for the symmetrized graph.
    #       These have to be determined in order to allocate space for the
    #       new (symmetrized) graph "h".
    #       The array "first" is a work vector; it marks the first element
    #       in each vertex list that has not yet been processed. Since it is
    #       assumed that the lists are in order, we can "march" through them,
    #       looking at each element only once. The complexity of this
    #       code is O(nEdges).
    # 
    c = zeros(IT, g.nv)
    first = zeros(IT, g.nv)
    c .= 0
    first .= g.xadj[1:g.nv]

    for i in 1:g.nv       # For each "column" list ...
        for j in first[i]:(g.xadj[i+1]-1)
            k = g.adj[j]
            c[i] = c[i] + 1

            if (k < i)             # "above" the diagonal ... end
                c[k] = c[k] + 1         # no element (i, k)
                h.nedges = h.nedges + 1 # in the lower triangle
            else                        # below the diagonal ...
                while (first[k] <= g.xadj[k+1])
                    if (first[k] == g.xadj[k+1])
                        # -  -  -  
                        #                           list is exhausted; no element (i, k)
                        #                           in the upper triangle.
                        # 
                        c[k] = c[k] + 1
                        h.nedges = h.nedges + 1
                        break
                    end
                    m = g.adj[first[k]]   # row number in column k
                    if (m < i)
                        #   end
                        #                           element is above i; no element (k, m)
                        #                           in the lower triangle. Update both counts
                        #                           and proceed to next element in the column.
                        # 
                        c[m] = c[m] + 1
                        h.nedges = h.nedges + 1
                        c[k] = c[k] + 1
                        first[k] = first[k] + 1
                    elseif (m == i)
                        #   end
                        #                           symmetric elements present; update column
                        #                           count for column k; no update for lower
                        #                           triangle needed; already recorded earlier.
                        #                           Suspend scan of column.
                        # 
                        c[k] = c[k] + 1
                        first[k] = first[k] + 1
                        break
                    else
                        # 
                        #                           element is below i; no element (i, k)
                        #                           in the upper triangle. Update column
                        #                           count; no count update for lower
                        #                           triangle needed; already recorded earlier.
                        #                           Suspend scan of column.
                        # 
                        c[k] = c[k] + 1
                        h.nedges = h.nedges + 1
                        break
                    end
                end # while
            end
        end
    end
    # 
    #       Allocate space for h now that nEdges is known.
    #       Compute xadj for h, and reset first for second pass through
    #       the original graph g.
    # 
    h.xadj = __extend(h.xadj, h.nv + 1)
    h.adj = __extend(h.adj, h.nedges)
    h.xadj[1] = 1

    for i in 1:g.nv
        first[i] = g.xadj[i]
        h.xadj[i+1] = h.xadj[i] + c[i]
        c[i] = h.xadj[i]
    end
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    #       Second pass ... is essentially identical to the first pass
    #       above, except we now have space to record the new edges,
    #       rather than just counting them.
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    for i in 1:g.nv       # For each "column" list ...
        for j = first[i]:(g.xadj[i+1]-1)
            k = g.adj[j]           # Add k to new list i
            h.adj[c[i]] = k
            c[i] = c[i] + 1
            if (k < i)        # "above" the diagonal ... 
                h.adj[c[k]] = i    # add i to new list k
                c[k] = c[k] + 1
            else                   # "below" the diagonal ...
                while (first[k] <= g.xadj[k+1])
                    if (first[k] == g.xadj[k+1])
                        # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                        #                           list is exhausted; missing element in
                        #                           the upper triangle. Add i to new list k.
                        # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                        h.adj[c[k]] = i
                        c[k] = c[k] + 1
                        break
                    end
                    m = g.adj[first[k]]  # row number in column k
                    if (m < i)
                        # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  end
                        #                           element is above i; missing element in
                        #                           the lower triangle. Update both new lists
                        #                           and proceed to next element in the column.
                        # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                        h.adj[c[m]] = k
                        c[m] = c[m] + 1
                        h.adj[c[k]] = m
                        c[k] = c[k] + 1
                        first[k] = first[k] + 1
                    elseif (m == i)
                        # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  end
                        #                           symmetric elements present; add i to new
                        #                           list k; no change in the lower
                        #                           triangle needed; already recorded earlier.
                        #                           Suspend scan of column.
                        # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                        h.adj[c[k]] = m
                        c[k] = c[k] + 1
                        first[k] = first[k] + 1
                        break
                    else
                        # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                        #                           element is below i; no element (i, k)
                        #                           in the upper triangle. Update new list k.
                        #                           no adjustment for lower
                        #                           triangle needed; already recorded earlier.
                        #                           Suspend scan of column.
                        # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                        h.adj[c[k]] = i
                        c[k] = c[k] + 1
                        break
                    end
                end
            end
        end
    end
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    #       Release storage for g, and  assign h to g.
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    g.nv = h.nv
    g.nedges = h.nedges
    g.nrows = h.nrows
    g.ncols = h.ncols
    g.xadj = h.xadj
    g.adj = h.adj
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    #       Sort the edges of the new graph in increasing order.
    #       By convention, graphs are maintained in this state for
    #       efficiency reasons.
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    sortgraph!(g)
    return true
end
 
"""
    sortgraph!(g::Graph{IT}) where {IT}

Sort the adjacency lists of the graph.

Important assumption:
This works only for graphs that are symmetric.
"""
function sortgraph!(g::Graph{IT}) where {IT}
    xadj = deepcopy(g.xadj)
    adj = zeros(IT, g.nedges)   
    for i in 1: g.nv
        for j in g.xadj[i]:(g.xadj[i + 1] - 1)
            k = g.adj[j];   adj[xadj[k]] = i
            xadj[k] = xadj[k] + 1
        end
    end
    g.adj .= adj
    return true
end

"""
    isstructuresymmetric(g::Graph{IT}) where {IT}

Determines if a graph is structurally symmetric.

Output: either true or false
"""
function isstructuresymmetric(g::Graph{IT}) where {IT}
    function findentry(i, j)
        for k = g.xadj[i]:g.xadj[i+1]-1
            if g.adj[k]==j
                return true
            end
        end
        return false
    end
    for i in 1: g.nv
        for k = g.xadj[i]:g.xadj[i+1]-1
            j = g.adj[k]
            if !findentry(j, i)
                return false
            end
        end
    end
    return true
end

end
