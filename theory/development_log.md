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

**Date:** 2026-07-03

## Goal

Extend the binary linear algebra library with additional matrix operations required for implementing Information Set Decoding algorithms.


## Implemented

### linear_algebra.py

- Matrix transposition
- Identity matrix generation

## Tested

### linear_algebra.py

#### Matrix transposition

- correct transposition
- empty matrix
- invalid matrix with rows of different lengths

#### Identity matrix

- correct identity matrix generation
- invalid matrix size

All tests passed.

## Notes

Matrix transposition was implemented explicitly using nested loops instead of Python shortcuts such as `zip(*matrix)`.
This makes the transformation easier to understand and keeps the implementation suitable for learning and later extension.

The identity matrix implementation will be used in future milestones, especially during Gaussian elimination and solving linear systems over GF(2).

---

## Next milestone

Continue implementing linear algebra utilities required by ISD algorithms:

- matrix-matrix multiplication over GF(2)
- Gaussian elimination over GF(2)
- matrix rank
- solving linear systems over GF(2)