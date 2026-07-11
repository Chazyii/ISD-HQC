# Development Log

---

# Milestone 1 — GF(2) Linear Algebra Foundations

**Date:** 2026-07-01

## Goal

Create the mathematical foundation for Information Set Decoding (ISD) algorithms by implementing the basic operations over GF(2).

## Implemented

### linear_algebra.py

- GF(2) vector addition
- Hamming weight
- GF(2) matrix-vector multiplication

## Tested

### linear_algebra.py

- vector addition
- invalid vector lengths
- Hamming weight
- matrix-vector multiplication
- empty matrix
- invalid dimensions

All tests passed.

## Notes

The project currently uses a simple Python representation:

- `Vector = list[int]`
- `Matrix = list[list[int]]`

This representation was intentionally chosen to keep the implementation simple and readable.
More efficient implementations (NumPy, bit arrays or C/C++) will be introduced after the correctness of the algorithms has been verified.

---

# Milestone 2 — Syndrome Decoding Primitives

**Date:** 2026-07-02

## Goal

Implement the basic building blocks required to formulate the Syndrome Decoding problem.

## Implemented

### syndrome.py

- Random error vector generation
- Random parity-check matrix generation
- Syndrome computation
- Solution verification

## Tested

### syndrome.py

#### Random error vector

- correct vector length
- correct Hamming weight
- binary values only
- invalid weight

#### Random parity-check matrix

- correct dimensions
- binary values only
- invalid row count
- invalid column count

#### Syndrome computation

- correct syndrome generation

#### Solution verification

- valid solution
- invalid syndrome
- valid weight
- invalid weight

All tests passed.

## Notes

The project is now able to generate complete random Syndrome Decoding instances:

- parity-check matrix `H`
- error vector `e`
- syndrome `s = He^T`

This provides the foundation required for implementing Information Set Decoding algorithms.


# Milestone 3 — Linear Algebra Utilities for ISD

**Date:** 2026-07-07

## Goal

Build the core linear algebra library over GF(2) required for implementing Information Set Decoding algorithms.

## Implemented

### linear_algebra.py

- Matrix transposition
- Identity matrix generation
- Matrix-matrix multiplication over GF(2)
- Gaussian elimination (row echelon form)
- Matrix rank computation

## Tested

### Matrix transposition

- correct transposition
- empty matrix
- invalid matrix with rows of different lengths

### Identity matrix

- correct identity matrix generation
- invalid matrix size

### Matrix-matrix multiplication

- correct multiplication
- incompatible matrix dimensions
- empty matrix

### Gaussian elimination

- correct row echelon form
- row swapping
- empty matrix
- invalid matrix representation

### Matrix rank

- full-rank matrix
- rank-deficient matrix
- zero matrix

All tests passed.

## Notes

The project now contains the complete set of fundamental linear algebra operations required by Information Set Decoding algorithms.

The implemented Gaussian elimination routine forms the basis for matrix rank computation and will also be used by subsequent ISD algorithms.

All operations are currently implemented using standard Python data structures to maximize readability and simplify debugging before future optimization.

---

# Milestone 4 — Prange Algorithm Foundations

**Date:** 2026-07-12

## Goal

Develop the fundamental building blocks required for the implementation of the Prange Information Set Decoding algorithm.

## Implemented

### algorithms/prange.py

- Random information set selection
- Construction of the induced linear system

### linear_algebra.py

- Gaussian elimination solver for linear systems over GF(2)

## Tested

### algorithms/prange.py

#### Information set selection

- correct information set size
- unique indices
- sorted indices
- indices within valid range
- invalid code length
- invalid code dimension

#### Induced linear system construction

- correct induced matrix
- correct complement construction
- syndrome preservation
- invalid parity-check matrix
- invalid syndrome length
- invalid information set size
- duplicate information set indices
- information set indices out of range

### linear_algebra.py

#### Linear system solver

- correct solution
- row swapping
- singular matrix
- inconsistent system
- empty matrix
- invalid matrix
- non-square matrix
- invalid vector length

All tests passed.

## Notes

The implementation now supports the first computational stage of the Prange algorithm.

Given a randomly selected information set, the corresponding induced linear system can be constructed by extracting the complement columns of the parity-check matrix.

A general solver for linear systems over GF(2) has also been implemented. This solver will be reused by the Prange algorithm and may also be employed by subsequent ISD algorithms.

The remaining implementation of Prange will build upon these components.

---

## Next milestone

Continue implementing the Prange algorithm:

- solve the induced linear system inside Prange
- reconstruct the candidate error vector
- verify the candidate solution
- implement the complete decoding loop