#include "add_mm.h"

#define INDEX_GETTER(width, height)                \
    int x = threadIdx.x + blockIdx.x * blockDim.x; \
    int y = threadIdx.y + blockIdx.y * blockDim.y; \
                                                   \
    if (!(x < width && y < height))                \
    {                                              \
        return;                                    \
    }                                            

__global__ void addMatricesByElements(const float *A, const float *B, float *C, int ncols, int nrows)
{
    INDEX_GETTER(ncols, nrows);


}

__global__ void addMatricesByRows(const float *A, const float *B, float *C, int ncols, int nrows)
{

}

__global__ void addMatricesByColumns(const float *A, const float *B, float *C, int ncols, int nrows)
{

}

Matrix addMatricesOnDevice(const Matrix &A, const Matrix &B, AddMethod method)
{

}

Matrix addMatricesOnHost(const Matrix &A, const Matrix &B)
{
    if (A.getRows() != B.getRows() || A.getCols() != B.getCols())
    {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }

    Matrix C(A.getRows(), A.getCols());
    for (unsigned int i = 0; i < A.getRows(); ++i)
    {
        for (unsigned int j = 0; j < A.getCols(); ++j)
        {
            C.getDataPtr()[i * A.getCols() + j] = A.getDataConstPtr()[i * A.getCols() + j] + B.getDataConstPtr()[i * A.getCols() + j];
        }
    }
    return C;
}
