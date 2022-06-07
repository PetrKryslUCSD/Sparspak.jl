var documenterSearchIndex = {"docs":
[{"location":"guide/guide.html","page":"Guide","title":"Guide","text":"Table of contents","category":"page"},{"location":"guide/guide.html#Guide","page":"Guide","title":"Guide","text":"","category":"section"},{"location":"index.html#Sparspak-Documentation","page":"Home","title":"Sparspak Documentation","text":"","category":"section"},{"location":"index.html#Package-features","page":"Home","title":"Package features","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Solves systems of coupled linear algebraic equations with a sparse coefficient matrix.\nReorderings of various kinds are supported, including the Multiple Minimum Degree (MMD).\nFactorizations of various kinds are supported.\nSolutions with multiple right hand sides, and solutions with preserved structure but changed matrix coefficients are supported. ","category":"page"},{"location":"index.html#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"The latest release of Sparspak can be installed from the Julia REPL prompt with","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"julia> ]add Sparspak","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"The closing square bracket switches to the package manager interface and the add commands installs Sparspak and any missing dependencies.  To return to the Julia REPL hit the delete key.","category":"page"},{"location":"index.html#Simple-example","page":"Home","title":"Simple example","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"This code makes up a random-coefficient (but diagonally dominant) sparse matrix and a simple right hand side vector. The sparse linear algebraic equation problem is then solved with the LU factorization. The solution is tested against the solution with the built-in solver.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"using Sparspak.Problem: Problem, insparse, outsparse, infullrhs\nusing Sparspak.SparseSolver: SparseSolver, solve\n\nfunction _test()\n    n = 1357\n    A = sprand(n, n, 1/n)\n    A = -A - A' + 20 * LinearAlgebra.I\n    \n    p = Problem(n, n)\n    insparse(p, A);\n    infullrhs(p, 1:n);\n    \n    s = SparseSolver(p)\n    solve(s, p)\n    A = outsparse(p)\n    x = A \\ p.rhs\n    @test norm(p.x - x) / norm(x) < 1.0e-6\n\n    return true\nend\n\n_test()","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"For more details see the file test/test_sparse_method.jl, module msprs016.","category":"page"},{"location":"index.html#User-guide","page":"Home","title":"User guide","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Pages = [\n    \"guide/guide.md\",\n]\nDepth = 1","category":"page"},{"location":"index.html#Manual","page":"Home","title":"Manual","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"The description of the types and the functions, organized by module and/or other logical principle.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Pages = [\n    \"man/types.md\",\n    \"man/functions.md\",\n]\nDepth = 3","category":"page"},{"location":"man/types.html#Types","page":"Types","title":"Types","text":"","category":"section"},{"location":"man/types.html#Problem","page":"Types","title":"Problem","text":"","category":"section"},{"location":"man/types.html","page":"Types","title":"Types","text":"Modules = [Sparspak.SpkProblem]\nPrivate = true\nOrder = [:type]","category":"page"},{"location":"man/types.html#Sparspak.SpkProblem.Problem-Union{Tuple{FT}, Tuple{IT}, Tuple{IT, IT}, Tuple{IT, IT, IT}, Tuple{IT, IT, IT, FT}, Tuple{IT, IT, IT, FT, Any}} where {IT, FT}","page":"Types","title":"Sparspak.SpkProblem.Problem","text":"\n\n\n\n","category":"method"},{"location":"man/types.html#Sparse-LU-Method","page":"Types","title":"Sparse LU Method","text":"","category":"section"},{"location":"man/types.html","page":"Types","title":"Types","text":"Modules = [Sparspak.SpkSparseSolver]\nPrivate = true\nOrder = [:type]","category":"page"},{"location":"man/types.html#Sparse-LU-SPD-Method","page":"Types","title":"Sparse LU SPD Method","text":"","category":"section"},{"location":"man/types.html","page":"Types","title":"Types","text":"Functions for Symmetric Positive Definite (SPD) matrices.","category":"page"},{"location":"man/types.html","page":"Types","title":"Types","text":"Modules = [Sparspak.SpkSparseSolver]\nPrivate = true\nOrder = [:type]","category":"page"},{"location":"man/types.html#Elimination-Trees","page":"Types","title":"Elimination Trees","text":"","category":"section"},{"location":"man/types.html","page":"Types","title":"Types","text":"Modules = [Sparspak.SpkETree]\nPrivate = true\nOrder = [:type]","category":"page"},{"location":"man/types.html#Sparspak.SpkETree.ETree-Tuple{IT} where IT","page":"Types","title":"Sparspak.SpkETree.ETree","text":"\n\n\n\n","category":"method"},{"location":"man/types.html#Graphs","page":"Types","title":"Graphs","text":"","category":"section"},{"location":"man/types.html","page":"Types","title":"Types","text":"Modules = [Sparspak.SpkGraph]\nPrivate = true\nOrder = [:type]","category":"page"},{"location":"man/types.html#Sparspak.SpkGraph.Graph-Union{Tuple{Sparspak.SpkProblem.Problem{IT}}, Tuple{IT}, Tuple{Sparspak.SpkProblem.Problem{IT}, Any}} where IT","page":"Types","title":"Sparspak.SpkGraph.Graph","text":"This routine constructs a graph from a problem object.\n\n\n\nIt does not check that the problem object contains a structurally   symmetric matrix, since sometimes only the lower or upper triangle of   a symmetric matrix may be stored. There are routines in this module to   make a given graph object structurally symmetric.\n\n\n\nInput:   g - the graph object, declared by the calling routine   p - the problem object, used to create the graph   diagonal - indicates that the diagonal elements are included. If     diagonal is not given, the adjacency structure does not include     the diagonal elements.   objectName - (optional) name to be assigned to g. Updated Parameter:    g - created graph object.\n\n\n\n\n\n","category":"method"},{"location":"man/types.html#Ordering","page":"Types","title":"Ordering","text":"","category":"section"},{"location":"man/types.html","page":"Types","title":"Types","text":"Modules = [Sparspak.SpkOrdering]\nPrivate = true\nOrder = [:type]","category":"page"},{"location":"man/types.html#Sparspak.SpkOrdering.Ordering-Tuple{IT} where IT","page":"Types","title":"Sparspak.SpkOrdering.Ordering","text":"ConstructOrdering constructs an ordering object. Since only one   parameter (nRows) is supplied, it is assumed that the size of the   row ordering and column ordering are the same. That is, that the   matrix is square. Input Parameters:   order - the ordering (declared in the calling program)   nRows - the number of rows (and columns) in the matrix   objectName - the name of the ordering object (optional) Output Parameter:    order - the updated ordering object\n\n\n\n\n\n","category":"method"},{"location":"man/types.html#Sparspak.SpkOrdering.Ordering-Union{Tuple{IT}, Tuple{IT, IT}} where IT","page":"Types","title":"Sparspak.SpkOrdering.Ordering","text":"ConstructOrdering2 constructs an ordering object. The arrays   rPerm, cPerm, rInvp, cInvp are allocated and initialized to   the identity permutation. Input Parameter:   order - the ordering (declared in the calling program)   nRows, nCols - the number of rows and columns in the matrix   objectName - the name of the ordering object (optional) Output Parameter:    order - the updated ordering object\n\n\n\n\n\n","category":"method"},{"location":"man/types.html#Grid","page":"Types","title":"Grid","text":"","category":"section"},{"location":"man/types.html","page":"Types","title":"Types","text":"Modules = [Sparspak.SpkGrid]\nPrivate = true\nOrder = [:type]","category":"page"},{"location":"man/types.html#Sparspak.SpkGrid.Grid-Union{Tuple{IT}, Tuple{IT, IT}} where IT","page":"Types","title":"Sparspak.SpkGrid.Grid","text":"\n\n\n\n","category":"method"},{"location":"man/types.html#Utilities","page":"Types","title":"Utilities","text":"","category":"section"},{"location":"man/types.html","page":"Types","title":"Types","text":"Modules = [Sparspak.SpkUtilities]\nPrivate = true\nOrder = [:type]","category":"page"},{"location":"man/functions.html#Functions","page":"Functions","title":"Functions","text":"","category":"section"},{"location":"man/functions.html#Problem","page":"Functions","title":"Problem","text":"","category":"section"},{"location":"man/functions.html","page":"Functions","title":"Functions","text":"Modules = [Sparspak.SpkProblem]\nPrivate = true\nOrder = [:function]","category":"page"},{"location":"man/functions.html#Sparspak.SpkProblem.infullrhs-Union{Tuple{FT}, Tuple{IT}, Tuple{Sparspak.SpkProblem.Problem{IT, FT}, Any}} where {IT, FT}","page":"Functions","title":"Sparspak.SpkProblem.infullrhs","text":"\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkProblem.makegridproblem-Union{Tuple{IT}, Tuple{IT, IT}} where IT","page":"Functions","title":"Sparspak.SpkProblem.makegridproblem","text":"This routine constructs a Grid object given an H and K, and fills in a   Problem object using this Grid. Input Parameters:   h - the number of rows in the Grid   k - the number of columns in the Grid   stencil - an optional variable specifying the difference operator             to be applied to the grid. Output Parameter:    p - the Problem object to be filled\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkProblem.makegridproblem-Union{Tuple{Sparspak.SpkGrid.Grid{IT}}, Tuple{IT}} where IT","page":"Functions","title":"Sparspak.SpkProblem.makegridproblem","text":"This routine fills in a Problem object using a given Grid. Input Parameters:   g - the Grid to be used to fill a Problem matrix   stencil - an optional variable specifying the difference operator             to be applied to the grid. Output Parameter:    p - the Problem object to be filled\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkProblem.makerhs-Union{Tuple{Sparspak.SpkProblem.Problem}, Tuple{FT}, Tuple{Sparspak.SpkProblem.Problem, Vector{FT}}, Tuple{Sparspak.SpkProblem.Problem, Vector{FT}, Any}} where FT","page":"Functions","title":"Sparspak.SpkProblem.makerhs","text":"This routine constructs the RHS of a problem given an x for the   equation Ax = rhs The x must have the same number of elements   as the problem (represented by A above) has columns   If x is not present  a right hand side is contructed so that   (a the) solution is 1 2 3 m Input Parameter   x - the vector in the equationAx = rhs\"\"   mType - matrix type (optional). If the matrix is symmetric and only             the lower or upper triangle is present, the user must let             the routine know this by setting mType to one of:                 \"L\" or \"l\" - when only the lower triangle is present                 \"U\" or \"u\" - when only the upper triangle is present                 \"T\" or \"t\" - when either the lower or upper triangle is                              present. Updated Parameter:    p - the problem for which the RHS is being constructed.\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparse-LU-Method","page":"Functions","title":"Sparse LU Method","text":"","category":"section"},{"location":"man/functions.html","page":"Functions","title":"Functions","text":"Modules = [Sparspak.SpkSparseSolver]\nPrivate = true\nOrder = [:function]","category":"page"},{"location":"man/functions.html#Sparspak.SpkSparseBase.factor-Union{Tuple{Sparspak.SpkSparseSolver.SparseSolver{IT}}, Tuple{IT}} where IT","page":"Functions","title":"Sparspak.SpkSparseBase.factor","text":"factor(s::SparseSolver{IT}) where {IT}\n\nNumerical factorization of the coefficient matrix.\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkSparseBase.findorder-Union{Tuple{F}, Tuple{IT}, Tuple{Sparspak.SpkSparseSolver.SparseSolver{IT}, F}} where {IT, F}","page":"Functions","title":"Sparspak.SpkSparseBase.findorder","text":"findorder(s::SparseSolver{IT}, orderfunction::F) where {IT, F}\n\nFind reordering of the coefficient matrix.\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkSparseBase.findorder-Union{Tuple{Sparspak.SpkSparseSolver.SparseSolver{IT}}, Tuple{F}, Tuple{IT}} where {IT, F}","page":"Functions","title":"Sparspak.SpkSparseBase.findorder","text":"findorder(s::SparseSolver{IT}) where {IT, F}\n\nFind reordering of the coefficient matrix using the default method.\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkSparseBase.inmatrix-Union{Tuple{IT}, Tuple{Sparspak.SpkSparseSolver.SparseSolver{IT}, Sparspak.SpkProblem.Problem{IT}}} where IT","page":"Functions","title":"Sparspak.SpkSparseBase.inmatrix","text":"inmatrix(s::SparseSolver{IT}, p::Problem{IT}) where {IT}\n\nPut numerical values of the matrix stored in the problem into the data structures of the solver.\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkSparseBase.symbolicfactor-Union{Tuple{Sparspak.SpkSparseSolver.SparseSolver{IT}}, Tuple{IT}} where IT","page":"Functions","title":"Sparspak.SpkSparseBase.symbolicfactor","text":"symbolicfactor(s::SparseSolver{IT})\n\nSymbolic factorization of the(reordered) matrix A and the creation of the data structures for the factorization and forward and backward substitution. \n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkSparseBase.triangularsolve-Union{Tuple{FT}, Tuple{IT}, Tuple{Sparspak.SpkSparseSolver.SparseSolver{IT, FT}, Vector{FT}}} where {IT, FT}","page":"Functions","title":"Sparspak.SpkSparseBase.triangularsolve","text":"triangularsolve(s::SparseSolver{IT, FT}, solution::Vector{FT}) where {IT, FT}\n\nForward and backward substitution (triangular solution).\n\nVariant where the right-hand side vector is passed in.\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkSparseBase.triangularsolve-Union{Tuple{IT}, Tuple{Sparspak.SpkSparseSolver.SparseSolver{IT}, Sparspak.SpkProblem.Problem{IT}}} where IT","page":"Functions","title":"Sparspak.SpkSparseBase.triangularsolve","text":"triangularsolve(s::SparseSolver{IT},  p::Problem{IT}) where {IT}\n\nForward and backward substitution (triangular solution).\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkSparseSolver.findorderperm-Union{Tuple{IT}, Tuple{Sparspak.SpkSparseSolver.SparseSolver{IT}, Any}} where IT","page":"Functions","title":"Sparspak.SpkSparseSolver.findorderperm","text":"findorderperm(s::SparseSolver{IT}, perm) where {IT}\n\nFind reordering of the coefficient matrix using a given permutation.\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkSparseSolver.solve-Union{Tuple{IT}, Tuple{Sparspak.SpkSparseSolver.SparseSolver{IT}, Sparspak.SpkProblem.Problem{IT}}} where IT","page":"Functions","title":"Sparspak.SpkSparseSolver.solve","text":"solve(s::SparseSolver{IT}, p::Problem{IT}) where {IT}\n\nExecute all the steps of the solution process:\n\nReordering of the matrix A \nSymbolic factorization of the(reordered) matrix A and the creation of the\n\ndata structures for the factorization and forward and backward substitution \n\nPutting numerical values of\n\nA into the data structures \n\nNumerical factorization of A \nForward and backward substitution (triangular solution) \n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Multiple-minimum-degree-(MMD)-ordering.","page":"Functions","title":"Multiple minimum degree (MMD) ordering.","text":"","category":"section"},{"location":"man/functions.html","page":"Functions","title":"Functions","text":"Modules = [Sparspak.SpkMmd]\nPrivate = true\nOrder = [:function]","category":"page"},{"location":"man/functions.html#Sparspak.SpkMmd.generalmmd-Union{Tuple{IT}, Tuple{IT, Vector{IT}, Vector{IT}, Vector{IT}, Vector{IT}}} where IT","page":"Functions","title":"Sparspak.SpkMmd.generalmmd","text":"generalmmd(n, xadj, adj, perm, invp)\n\nThis routine implements the minimum degree algorithm.  It makes use of the implicit representation of elimination graphs by quotient graphs, and the notion of indistinguishable nodes.  It also implements the modifications by multiple elimination and minimum external degree.\n\nInput parameters -\n\nn - number of equations (xadj, adj) - adjacency structure for the graph. delta - tolerance value for multiple elimination. FIX ME: should delta be passed as argument?\n\nOutput:   perm, invp - the minimum degree Ordering.\n\nWorking arrays -   degHead (deg) - points to first node with degree deg, or 0 if there                are no such nodes.   degNext (node) - points to the next node in the degree list                associated with node, or 0 if node was the last in the                degree list.   degPrev (node) - points to the previous node in a degree list                associated with node, or the negative of the degree of                node (if node was the last in the degree list), or 0                if the node is not in the degree lists.   superSIze - the size of the supernodes.   elimHead - points to the first node eliminated in the current pass                Using elimNext, one can determine all nodes just                eliminated.   elimNext (node) - points to the next node in a eliminated supernode                or 0 if there are no more after node.   marker - marker vector.   mergeParent - the parent map for the merged forest.     needsUpdate (node) - > 0 iff node needs degree update.(0 otherwise)\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkMmd.mmdelim-NTuple{13, Any}","page":"Functions","title":"Sparspak.SpkMmd.mmdelim","text":"Purpose - This routine eliminates the node mdNode of\n  minimum degree from the adjacency structure, which\n  is stored in the quotient Graph format.  It also\n  transforms the quotient Graph representation of the\n  elimination Graph.\n\nInput parameters -       mdNode - node of minimum degree.       tag - tag value.       invp - the inverse of an incomplete minimum degree Ordering.                 (It is zero at positions where the Ordering is unknown.)    Updated parameters -       (xadj, adjncy) - updated adjacency structure (xadj is not updated).       degHead (deg) - points to first node with degree deg, or 0 if there                    are no such nodes.       degNext (node) - points to the next node in the degree list                    associated with node, or 0 if node was the last in the                    degree list.       degPrev (node) - points to the previous node in a degree list                    associated with node, or the negative of the degree of                    node (if node was the last in the degree list), or 0                    if the node is not in the degree lists.       superSIze - the size of the supernodes.       elimNext (node) - points to the next node in a eliminated supernode                    or 0 if there are no more after node.       marker - marker vector.       mergeParent - the parent map for the merged forest.        needsUpdate (node) - > 0 iff node needs update. (0 otherwise)\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkMmd.mmdupdate-NTuple{16, Any}","page":"Functions","title":"Sparspak.SpkMmd.mmdupdate","text":"Purpose - This routine updates the degrees of nodes\n  after a multiple elimination step.\n\nInput parameters -       elimHead - the beginning of the list of eliminated                nodes (i.e., newly formed elements).       neqns - number of equations.       (xadj, adjncy) - adjacency structure.       delta - tolerance value for multiple elimination.       invp - the inverse of an incomplete minimum degree Ordering.                (It is zero at positions where the Ordering is unknown.)    Updated parameters -       mindeg - new minimum degree after degree update.       degHead (deg) - points to first node with degree deg, or 0 if there                    are no such nodes.       degNext (node) - points to the next node in the degree list                    associated with node, or 0 if node was the last in the                    degree list.       degPrev (node) - points to the previous node in a degree list                    associated with node, or the negative of the degree of                    node (if node was the last in the degree list), or 0                    if the node is not in the degree lists.       superSIze - the size of the supernodes.       elimNext (node) - points to the next node in a eliminated supernode                    or 0 if there are no more after node.       marker - marker vector for degree update.       tag - tag value.       mergeParent - the parent map for the merged forest.        needsUpdate (node) - > 0 iff node needs update. (0 otherwise)\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Symbolic-Factorization","page":"Functions","title":"Symbolic Factorization","text":"","category":"section"},{"location":"man/functions.html","page":"Functions","title":"Functions","text":"Modules = [Sparspak.SpkSymfct]\nPrivate = true\nOrder = [:function]","category":"page"},{"location":"man/functions.html#Sparspak.SpkSymfct.findcolumncounts-NTuple{8, Any}","page":"Functions","title":"Sparspak.SpkSymfct.findcolumncounts","text":"This subroutine determines the column counts in\nthe Cholesky factor.  It uses an algorithm due to Joseph Liu\nfound in SIMAX 11, 1990, pages 144 - 145.0\n\nInput parameters:     (i) n - number of equations.     (i) xadj - array of length n + 1, containing pointers                         to the adjacency structure.     (i) adj - array of length xadj(n + 1) - 1, containing                         the adjacency structure.     (i) perm - array of length n, containing the                         postordering.     (i) invp - array of length n, containing the                         inverse of the postordering.     (i) parent - array of length n, containing the                         elimination tree of the postordered matrix.\n\nOutput parameters:     (i) colcnt - array of length n, containing the number                         of nonzeros in each column of the factor,                         including the diagonal entry.     (i) nlnz - number of nonzeros in the factor, including                         the diagonal entries.\n\nWork parameters:     (i) marker - array of length n used to mark the                          vertices visited in each row subtree.\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkSymfct.findnumberofsupermods-NTuple{7, Any}","page":"Functions","title":"Sparspak.SpkSymfct.findnumberofsupermods","text":"\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkSymfct.findschedule-NTuple{6, Any}","page":"Functions","title":"Sparspak.SpkSymfct.findschedule","text":"\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkSymfct.findsupernodes-NTuple{8, Any}","page":"Functions","title":"Sparspak.SpkSymfct.findsupernodes","text":"\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkSymfct.findsupernodetree-NTuple{6, Any}","page":"Functions","title":"Sparspak.SpkSymfct.findsupernodetree","text":"\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkSymfct.symbolicfact-NTuple{12, Any}","page":"Functions","title":"Sparspak.SpkSymfct.symbolicfact","text":"This routine performs supernodal symbolic\nfactorization on a reordered linear system.\nThis is essentially a Fortran 90 translation of a code written\nby Esmond Ng and Barry Peyton.\n\nInput parameters:     (i) n - number of equations     (i) xadj - array of length n + 1 containing pointers                         to the adjacency structure.     (i) adj - array of length xadj(n + 1) - 1 containing                         the adjacency structure.     (i) perm - array of length n containing the                         postordering.     (i) invp - array of length n containing the                         inverse of the postordering.     (i) colcnt - array of length n, containing the number                         of nonzeros (non - empty rows) in each                         column of the factor,                         including the diagonal entry.     (i) nsuper - number of supernodes.     (i) xsuper - array of length nsuper + 1, containing the                         first column of each supernode.     (i) snode - array of length n for recording                         supernode membership.     (i) nofsub - number of subscripts to be stored in                         lindx.\n\nOutput parameters:     (i) xlindx - array of length n + 1, containing pointers                         into the subscript vector.     (i) lindx - array of length maxsub, containing the                         compressed subscripts.     (i) xlnz - column pointers for l.\n\nWorking parameters:     (i) mrglnk - array of length nsuper, containing the                         children of each supernode as a linked list.     (i) rchlnk - array of length n + 1, containing the                         current linked list of merged indices (the                         \"reach\" set).     (i) marker - array of length n used to mark indices                         as they are introduced into each supernode\"s                         index set.\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Elimination-Trees","page":"Functions","title":"Elimination Trees","text":"","category":"section"},{"location":"man/functions.html","page":"Functions","title":"Functions","text":"Modules = [Sparspak.SpkETree]\nPrivate = true\nOrder = [:function]","category":"page"},{"location":"man/functions.html#Sparspak.SpkETree.binarytree-NTuple{4, Any}","page":"Functions","title":"Sparspak.SpkETree.binarytree","text":"\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkETree.findetree-NTuple{6, Any}","page":"Functions","title":"Sparspak.SpkETree.findetree","text":"\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkETree.getetree-Tuple{Sparspak.SpkGraph.Graph, Sparspak.SpkOrdering.Ordering, Sparspak.SpkETree.ETree}","page":"Functions","title":"Sparspak.SpkETree.getetree","text":"Given a graph and an ordering, GetETree finds the corresponding   elimination tree. It calls the subroutine FindETree, which actually   does the work.\n\nInput Parameter:   g - the graph whose elimination tree is to be found.   order - the ordering for g\n\nUpdated Parameters:    t - the elimination tree.\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkETree.getpostorder-Union{Tuple{IT}, Tuple{Sparspak.SpkETree.ETree{IT}, Sparspak.SpkOrdering.Ordering, Any}} where IT","page":"Functions","title":"Sparspak.SpkETree.getpostorder","text":"\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkETree.postordertree-NTuple{6, Any}","page":"Functions","title":"Sparspak.SpkETree.postordertree","text":"\n\n\n\n","category":"method"},{"location":"man/functions.html#Sparspak.SpkETree.weightedbinarytree-NTuple{5, Any}","page":"Functions","title":"Sparspak.SpkETree.weightedbinarytree","text":"\n\n\n\n","category":"method"},{"location":"man/functions.html#Graphs","page":"Functions","title":"Graphs","text":"","category":"section"},{"location":"man/functions.html","page":"Functions","title":"Functions","text":"Modules = [Sparspak.SpkGraph]\nPrivate = true\nOrder = [:function]","category":"page"},{"location":"man/functions.html#Sparspak.SpkGraph.sortgraph-Union{Tuple{Sparspak.SpkGraph.Graph{IT}}, Tuple{IT}} where IT","page":"Functions","title":"Sparspak.SpkGraph.sortgraph","text":"SortGraph - sort the adjacency lists of the graph Important assumption:   This works only for graphs that are symmetric.  Output: updated graph\n\n\n\n\n\n","category":"method"},{"location":"man/functions.html#Ordering","page":"Functions","title":"Ordering","text":"","category":"section"},{"location":"man/functions.html","page":"Functions","title":"Functions","text":"Modules = [Sparspak.SpkOrdering]\nPrivate = true\nOrder = [:function]","category":"page"},{"location":"man/functions.html#Grid","page":"Functions","title":"Grid","text":"","category":"section"},{"location":"man/functions.html","page":"Functions","title":"Functions","text":"Modules = [Sparspak.SpkGrid]\nPrivate = true\nOrder = [:function]","category":"page"},{"location":"man/functions.html#Utilities","page":"Functions","title":"Utilities","text":"","category":"section"},{"location":"man/functions.html","page":"Functions","title":"Functions","text":"Modules = [Sparspak.SpkUtilities]\nPrivate = true\nOrder = [:function]","category":"page"},{"location":"man/functions.html#Sparspak.SpkUtilities.extend","page":"Functions","title":"Sparspak.SpkUtilities.extend","text":"extend(v::Vector, newlen::Integer, flagval=zero(eltype(v)))\n\nChange the size (smaller or larger).    \n\nThe contents of the vectors / arrays are preserved.\nThe changes in size may be positive or negative.\nflagval is the initialization value of the new parts of the arrays.   If it is absent,  the default is zero (for numerical arrays).\n\n\n\n\n\n","category":"function"},{"location":"man/functions.html#Sparspak.SpkUtilities.extend-2","page":"Functions","title":"Sparspak.SpkUtilities.extend","text":"extend(v::Matrix, newrow::Integer, newcol::Integer, flagval=zero(eltype(v)))\n\nChange the size of a matrix (smaller or larger).\n\n\n\n\n\n","category":"function"}]
}
