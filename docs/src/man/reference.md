# Types

## Problem

```@meta
CurrentModule = Sparspak.SpkProblem
```

```@docs
SpkProblem
```

## Sparse LU Solver

```@meta
CurrentModule = Sparspak.SpkSparseSolver
```

```@docs
SpkSparseSolver
```

## Elimination Trees

```@autodocs
Modules = [Sparspak.SpkETree]
Private = true
Order = [:type]
```

## Graphs

```@autodocs
Modules = [Sparspak.SpkGraph]
Private = true
Order = [:type]
```

## Ordering

```@autodocs
Modules = [Sparspak.SpkOrdering]
Private = true
Order = [:type]
```

## Grid

```@autodocs
Modules = [Sparspak.SpkGrid]
Private = true
Order = [:type]
```

# Functions

## Problem

```@meta
CurrentModule = Sparspak.SpkProblem
```

```@autodocs
Problem
inaij
inbi
insparse
infullrhs
outsparse
computeresidual
makegridproblem
makerhs
```

## Graphs

```@autodocs
Modules = [Sparspak.SpkGraph]
Private = true
Order = [:function]
```

## Ordering

```@autodocs
Modules = [Sparspak.SpkOrdering]
Private = true
Order = [:function]
```

## Multiple minimum degree (MMD) ordering.

```@autodocs
Modules = [Sparspak.SpkMmd]
Private = true
Order = [:function]
```

## Elimination Trees

```@autodocs
Modules = [Sparspak.SpkETree]
Private = true
Order = [:function]
```

## Symbolic Factorization

```@autodocs
Modules = [Sparspak.SpkSymfct]
Private = true
Order = [:function]
```

## Grid

```@autodocs
Modules = [Sparspak.SpkGrid]
Private = true
Order = [:function]
```
