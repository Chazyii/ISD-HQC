# Development Log

---

# Milestone 1 — GF(2) Linear Algebra Foundations

**Date:** 2026-07-01

## Goal

Create the mathematical foundation for Information Set Decoding (ISD) algorithms by implementing the basic operations over GF(2).

## Theory learned

- Arithmetic over GF(2)
- Binary vectors
- Binary matrices
- Hamming weight
- Matrix-vector multiplication over GF(2)

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

## Theory learned

- Error vectors
- Parity-check matrices
- Syndrome computation
- Solution verification
- Random instance generation for Syndrome Decoding

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

---

## Next milestone

Implement linear algebra utilities required by ISD algorithms:

- matrix transposition
- Gaussian elimination over GF(2)
- matrix rank
- solving linear systems over GF(2)

# Milestone 3 — Linear Algebra Utilities for ISD

**Date:** 2026-07-02

## Goal

Implement additional matrix operations required by Information Set Decoding algorithms.

## Theory learned

- Matrix transposition
- Relationship between matrix rows and columns
- Matrix dimension validation
- Handling invalid matrix representations

## Implemented

### linear_algebra.py

- Matrix transposition

## Tested

### linear_algebra.py

- matrix transposition
- empty matrix
- invalid matrix with rows of different lengths

All tests passed.

## Notes

Matrix transposition was implemented explicitly using nested loops instead of Python shortcuts such as `zip(*matrix)`.
This makes the transformation easier to understand and keeps the implementation suitable for learning and later extension.

---

## Next milestone

Continue implementing linear algebra utilities required by ISD algorithms:

- identity matrix
- matrix-matrix multiplication over GF(2)
- Gaussian elimination over GF(2)
- matrix rank
- solving linear systems over GF(2)