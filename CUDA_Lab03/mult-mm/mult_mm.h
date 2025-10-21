#ifndef MULT_MM_H_
#define MULT_MM_H_

#include <matrix.h>

#define TILE_SIZE 16

/**
 * @brief Enumeration for different multiplication methods
 *
 */
enum class MultMethod
{
    Standard,
    Tiled,
    Granular
};

/**
 * @brief Multiply two matrices using host (CPU)
 *
 * This function multiplies two matrices using standard CPU computation.
 * @param A First input matrix
 * @param B Second input matrix
 * @return Matrix Resulting matrix
 */
Matrix multMatrixMatrixOnHost(const Matrix &A, const Matrix &B);

/**
 * @brief Multiply two matrices using parallel computing
 *
 * This function multiplies two matrices by mapping one thread into one element of the resulting matrix.
 * @param A First input matrix
 * @param B Second input matrix
 * @param method Multiplication method to use (default: Standard)
 * @return Matrix Resulting matrix
 */
Matrix multMatrixMatrixOnDevice(const Matrix &A, const Matrix &B, MultMethod method = MultMethod::Standard);

#endif /* MULT_MM_H_ */