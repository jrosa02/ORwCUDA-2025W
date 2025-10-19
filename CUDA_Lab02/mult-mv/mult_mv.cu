#include "mult_mv.h"

#define GET_IDX_ROW_MAJOR(width, x, y) (((y) * (width)) + (x))

__global__ void multMatrixVector(float *b, float *A, float *x, unsigned int nrows, unsigned int ncols)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (!(row < nrows))
    {
        return;
    }

    float b_accum = 0;

    for (size_t col = 0; col < ncols; col++)
    {
        b_accum = A[GET_IDX_ROW_MAJOR(ncols, col, row)]*x[row];
    }
    
    b[row] = b_accum;
}

Matrix multMatrixVectorOnDevice(const Matrix &A, const Matrix &x)
{
    Matrix b = Matrix(x);

    // allocate input and output images in the device
    float *d_A;
    float *d_x;
    float *d_b;
    cudaMalloc((void **)&d_A, A.getRows() * A.getCols() * sizeof(float));
    cudaMalloc((void **)&d_x, x.getRows() * x.getCols() * sizeof(float));
    cudaMalloc((void **)&d_b, b.getRows() * b.getCols() * sizeof(float));

    // copy image to the device
    cudaMemcpy(d_A, A.getDataConstPtr(), A.getRows() * A.getCols() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, A.getDataConstPtr(), x.getRows() * x.getCols() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil((float)A.getRows()), 1);
    dim3 dimBlock(1, 1, 1);
    multMatrixVector<<<dimGrid, dimBlock>>>(d_b, d_A, d_x, A.getRows(), A.getCols());

    cudaMemcpy(b.getDataPtr(), d_b, b.getRows() * b.getCols() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);

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
