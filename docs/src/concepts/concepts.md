# Concepts

## Performance

Some preliminary data was collected for one particular sparse matrix system, 
with a symmetric matrix,  63070 equations, 4.22 million non-zeros.

On a Surface Pro 7 with 16 GB of RAM, i7-1065G7 @ 1.30 GHz, the following results were gathered under Windows 10.

| :-----: | :-----: | :-----: |
|         | Without MKL | With MKL |
| :-----: | :-----: | :-----: |
| Sparspak | 31   |    21    |
| UMFPACK  |    22   |   19 |
| :-----: | :-----: | :-----: |

On the same machine with Windows Subsystem for Linux, WSL 2, running Ubuntu 22.04, the results were as

| :-----: | :-----: | :-----: |
|         | Without MKL | With MKL |
| :-----: | :-----: | :-----: |
| Sparspak | 39   |    19    |
| UMFPACK  |    19   |   16 |
| :-----: | :-----: | :-----: |

