""" Graph class:
#
nV - the number of vertices in the graph.
(xadj, adj) - array pair storing the adjacency lists of the vertices.

The adjacency lists of the graph are stored in consecutive locations
in the array adj. The adjacency list for the i - th vertex in the graph
is stored in positions adj(k), k = xadj(i), .... xadj(i + 1) - 1.
#
When the graph is symmetric, if vertex i is in vertex j"s adjacency
 vertex j is in vertex i"s list. Using the representation above
each edge in the graph is stored twice.

There are no self - loops (No "diagonal elements") by default. If
diagonal elements are required, just type "diagonal" or "diag"
as an input parameter to the Construct subroutine.
#
For convenience in accessing the lists, xadj is of length nV + 1, with
xadj(nV + 1) = nEdges + 1.0 Thus, accessing vertex nV"s list is the
same as for any other of the vertices.

Graphs are created from Problem objects, which have a certain number
of rows (nRows) and columns (nCols). These numbers are captured
and stored in Graph objects as nRows and nCols as well.
"""
module SpkGraph

using ..SpkProblem: Problem

mutable struct Graph{IT}
    nv::IT
    nedges::IT
    nrows::IT
    ncols::IT
    xadj::Vector{IT}
    adj::Vector{IT}
end

"""
  This routine constructs a graph from a problem object.
#
  It does not check that the problem object contains a structurally
  symmetric matrix, since sometimes only the lower or upper triangle of
  a symmetric matrix may be stored. There are routines in this module to
  make a given graph object structurally symmetric.
#
Input:
  g - the graph object, declared by the calling routine
  p - the problem object, used to create the graph
  diagonal - indicates that the diagonal elements are included. If
    diagonal is not given, the adjacency structure does not include
    the diagonal elements.
  objectName - (optional) name to be assigned to g.
Updated Parameter:
   g - created graph object.
"""
    function Graph(p::Problem{IT}, diagonal=false) where {IT}
        # type (graph) ::  g
        # type (problem):: p
        # character (len = *), optional :: diagonal, objectname
        # character (len = *), parameter :: fname = "constructgraph:"
        # integer :: i, j, k, ptr
    
        nv = p.ncols
        nrows = p.nrows
        ncols = p.ncols
        if (diagonal)
            nedges = p.nedges
        else
            nedges = p.nedges - p.dedges
        end
        xadj = fill(zero(IT), nv + 1)
        adj = fill(zero(IT), nedges)
    
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

end