#ifndef MULT_MV_H_
#define MULT_MV_H_

#include <matrix.h>

/**
 * @brief Multiply matrix and vector using host (CPU)
 *
 * This function multiplies a matrix and a vector using standard CPU computation.
 * @param A Input matrix
 * @param x Input vector (as a matrix with one column)
 * @return Matrix Resulting vector (as a matrix with one column)
 */
Matrix multMatrixVectorOnHost(const Matrix &A, const Matrix &x);

/**
 * @brief Multiply matrix and vector using parallel computing
 *
 * This function multiplies a matrix and a vector by mapping one thread into one row of the resulting vector.
 * @param A Input matrix
 * @param x Input vector (as a matrix with one column)
 * @return Matrix Resulting vector (as a matrix with one column)
 */
Matrix multMatrixVectorOnDevice(const Matrix &A, const Matrix &x);

#endif /* MULT_MV_H_ */