# Concepts

## Performance

To compare the sparse LU factorization we can consider the UMFPACK `lu`
factorization from SuiteSparse, and the general LU sparse matrix factorization
of Sparspak. The former is a multi-frontal algorithm, the latter is a
super-nodal algorithm. Both rely quite heavily on dense algebra subroutines.

Some preliminary data was collected for one particular sparse matrix system, 
with a symmetric matrix,  63070 equations, 4.22 million non-zeros.

On a Surface Pro 7 with 16 GB of RAM, i7-1065G7 @ 1.30 GHz, the following results were gathered under Windows 10.


|         | UMFPACK | Sparspak |
| :---: | :---: | :---: |
| Without MKL [sec] | 22   |    31    |
| With MKL [sec] |    19   |   21 |

On the same machine with Windows Subsystem for Linux, WSL 2, running Ubuntu 22.04, the results were as


|         | UMFPACK | Sparspak |
| :---: | :---: | :---: |
| Without MKL  [sec] | 19   |    39    |
| With MKL  [sec] |    16   |   19 |

Clearly, MKL can make a huge difference 

