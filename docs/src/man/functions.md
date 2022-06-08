# Functions

## Problem

```@meta
CurrentModule = Sparspak.SpkProblem
```

```@docs
Problem(nrows::IT, ncols::IT, nnz::IT=2500, z::FT=0.0, info = "") where {IT, 
    FT}
inaij!
inbi!
insparse!
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
Order = [:function]
```

## Graphs

```@docs
Modules = [Sparspak.SpkGraph]
Private = true
Order = [:function]
```

## Ordering

```@docs
Modules = [Sparspak.SpkOrdering]
Private = true
Order = [:function]
```

## Multiple minimum degree (MMD) ordering.

```@docs
Modules = [Sparspak.SpkMmd]
Private = true
Order = [:function]
```

## Elimination Trees

```@docs
Modules = [Sparspak.SpkETree]
Private = true
Order = [:function]
```

## Symbolic Factorization

```@docs
Modules = [Sparspak.SpkSymfct]
Private = true
Order = [:function]
```

## Grid

```@docs
Modules = [Sparspak.SpkGrid]
Private = true
Order = [:function]
```
