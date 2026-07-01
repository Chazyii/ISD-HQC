# Development Log

---

## Milestone 1
Date: 2026-07-01

### Goal

Create the mathematical foundation for ISD algorithms and implement the first utilities for binary linear algebra.

### Theory learned

- Arithmetic over GF(2)
- Binary vectors and matrices
- Hamming weight
- Matrix-vector multiplication over GF(2)
- Error vectors in code-based cryptography

### Implemented

#### linear_algebra.py

- GF(2) vector addition
- Hamming weight
- GF(2) matrix-vector multiplication

#### syndrome.py

- Random error vector generation

### Tested

#### linear_algebra.py

- vector addition
- invalid vector lengths
- Hamming weight
- matrix-vector multiplication
- empty matrix
- invalid dimensions

#### syndrome.py

- correct vector length
- correct Hamming weight
- binary values only
- invalid weight

All tests passed.

### Notes

The project currently uses a simple Python representation:

- Vector = list[int]
- Matrix = list[list[int]]

This representation was intentionally chosen to keep the implementation easy to understand.
More efficient representations (NumPy, bit arrays or C/C++) will be introduced later when implementing ISD algorithms.

### Next milestone

Implement Syndrome Decoding primitives:

- compute syndrome
- verify solution
- generate random parity-check matrix