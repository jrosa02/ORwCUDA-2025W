#include "mult_mv.h"

#define GET_IDX_ROW_MAJOR(width, x, y) (((y) * (width)) + (x))

__global__ void multMatrixVector(float *b, const float *A, const float *x,
                                 unsigned int nrows, unsigned int ncols)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= nrows) return;

    float b_accum = 0.0f;

    // Multiply each element in the row of A by the corresponding element in x
    for (unsigned int col = 0; col < ncols; ++col)
    {
        b_accum += A[GET_IDX_ROW_MAJOR(ncols, col, row)] * x[col];
    }

    // Debug print (optional)
    // printf("b[%d] = %f\n", row, b_accum);

    b[row] = b_accum;
}

Matrix multMatrixVectorOnDevice(const Matrix &A, const Matrix &x)
{
    if (A.getCols() != x.getRows())
    {
        throw std::runtime_error("Matrix and vector dimensions do not match for multiplication.");
    }

    Matrix b(A.getRows(), 1);

    // Device memory
    float *d_A = nullptr;
    float *d_x = nullptr;
    float *d_b = nullptr;

    size_t sizeA = A.getRows() * A.getCols() * sizeof(float);
    size_t sizeX = x.getRows() * x.getCols() * sizeof(float);
    size_t sizeB = b.getRows() * b.getCols() * sizeof(float);

    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_x, sizeX);
    cudaMalloc((void **)&d_b, sizeB);

    // Copy data to device
    cudaMemcpy(d_A, A.getDataConstPtr(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.getDataConstPtr(), sizeX, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (A.getRows() + threadsPerBlock - 1) / threadsPerBlock;

    printf("Entering Kernel\n");
    multMatrixVector<<<blocksPerGrid, threadsPerBlock>>>(d_b, d_A, d_x, A.getRows(), A.getCols());
    cudaDeviceSynchronize();  // Ensure kernel completes before copying
    printf("Leaving Kernel\n");

    // Copy result back
    cudaMemcpy(b.getDataPtr(), d_b, sizeB, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_b);

    return b;
}

Matrix multMatrixVectorOnHost(const Matrix &A, const Matrix &x)
{
    if (A.getCols() != x.getRows())
    {
        throw std::runtime_error("Matrix and vector dimensions do not match for multiplication.");
    }

    Matrix b(A.getRows(), 1);
    for (unsigned int i = 0; i < A.getRows(); ++i)
    {
        float sum = 0.0f;
        for (unsigned int j = 0; j < A.getCols(); ++j)
        {
            sum += A.getDataConstPtr()[i * A.getCols() + j] * x.getDataConstPtr()[j];
        }
        b.getDataPtr()[i] = sum;
    }
    return b;
}
