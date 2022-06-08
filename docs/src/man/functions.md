# Functions

## Problem

```@meta
CurrentModule = Sparspak.SpkProblem
```

```@docs
Problem(nrows::IT, ncols::IT, nnz::IT=2500, z::FT=0.0, info = "") where {IT, 
    FT}
inaij!(p::Problem{IT,FT}, rnum, cnum, aij=zero(FT)) where {IT,FT}
inbi!(p::Problem{IT, FT}, rnum::IT, bi::FT) where {IT, FT}
insparse!(p::Problem{IT,FT}, spm) where {IT,FT}
infullrhs!
outsparse
computeresidual
makerhs!
makegridproblem
```

## Sparse LU Solver

```@meta
CurrentModule = Sparspak.SpkSparseSolver
```

```@docs
SparseSolver
findorder!
findorderperm!
symbolicfactor!
inmatrix!
factor!
triangularsolve!
solve!
```

## Multiple minimum degree (MMD) ordering.

```@meta
CurrentModule = Sparspak.SpkMMD
```

```@docs
mmd!(g::Graph, order::Ordering)
```

## Graphs

```@meta
CurrentModule = Sparspak.SpkGraph
```

```@docs
Graph(p::Problem{IT}, diagonal=false) where {IT}
makestructuresymmetric(g::Graph{IT}) where {IT}
sortgraph!(g::Graph{IT}) where {IT}
isstructuresymmetric(g::Graph{IT}) where {IT}
```

## Grid

```@meta
CurrentModule = Sparspak.SpkGrid
```

```@docs
Grid(h::IT, k::IT) where {IT}
```
