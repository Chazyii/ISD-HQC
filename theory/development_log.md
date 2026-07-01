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