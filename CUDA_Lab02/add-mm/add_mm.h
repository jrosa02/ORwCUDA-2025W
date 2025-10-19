#ifndef ADD_MM_H_
#define ADD_MM_H_

#include <matrix.h>

/**
 * @brief Enumeration for different addition methods
 *
 */
enum class AddMethod
{
    ByElements,
    ByRows,
    ByColumns
};

/**
 * @brief Add two matrices using parallel computing
 *
 * This function adds two matrices by mapping one thread into one row of the resulting matrix.
 * @param A First matrix
 * @param B Second matrix
 * @return Matrix Resulting matrix
 */
Matrix addMatricesOnDevice(const Matrix &A, const Matrix &B, AddMethod method);

/**
 * @brief Add two matrices using host (CPU)
 *
 * This function adds two matrices using standard CPU computation.
 * @param A First matrix
 * @param B Second matrix
 * @return Matrix Resulting matrix
 */
Matrix addMatricesOnHost(const Matrix &A, const Matrix &B);

#endif /* ADD_MM_H_ */