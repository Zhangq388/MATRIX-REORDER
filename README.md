# Locality-Activated Matrix Reordering Framework for SpMV

This repository contains the source code for the paper:

**A Locality-Activated Matrix Reordering Framework for SpMV**

The project includes the implementation of the RD-based headroom predictor and the pGOrder matrix reordering algorithm for sparse matrix-vector multiplication (SpMV).

## Structure

- `src/`: source files
- `include/`: header files
- `config/`: configuration files
- `ridge/`: predictor-related code
- `rcm/`, `rabbitorder/`, `parsb/`: baseline or comparison methods
- `test_rmat_csr.txt`, `test_rmat_meribit.txt`: experimental records

## Build

```bash
mkdir -p build
cd build
cmake ..
make -j
