#include "mult_mm.h"

__device__ __shared__

#define TILE_SIZE 16
#define GET_IDX_ROW_MAJOR(width, x, y) (((y) * (width)) + (x))
#define CHECK_MEM_RANGE_2D(index, ncols, nrows) if((index) >= ((ncols) * (nrows))) return;

__global__ void matrixMulKernel(const float *A, const float *B, float *C,
                                int A_rows, int A_cols, int B_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds to avoid illegal memory access
    if (row < A_rows && col < B_cols)
    {
        float accumulator = 0.0f;

        for (int k = 0; k < A_cols; ++k)
        {
            // A is A_rows x A_cols
            // B is A_cols x B_cols
            float a = A[row * A_cols + k];
            float b = B[k * B_cols + col];
            accumulator += a * b;
        }

        // C is A_rows x B_cols
        C[row * B_cols + col] = accumulator;
    }
}


__global__ void matrixMulTiledKernel(const float *A, const float *B, float *C,
                                     int A_rows, int A_cols, int B_cols)
{
}

__global__ void matrixMulGranularKernel(const float *A, const float *B, float *C,
                                        int A_rows, int A_cols, int B_cols)
{
}

Matrix multMatrixMatrixOnDevice(const Matrix &A, const Matrix &B, MultMethod method)
{
    // Fixed: Check only A and B dimensions
    if (A.getCols() != B.getRows())
    {
        throw std::runtime_error("Matrix dimensions do not match for addition.");
    }

    Matrix C(A.getRows(), B.getCols());

    // Device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    size_t A_size = A.getRows() * A.getCols() * sizeof(float);
    size_t B_size = B.getRows() * B.getCols() * sizeof(float);
    size_t C_size = C.getRows() * C.getCols() * sizeof(float);

    cudaMalloc((void **)&d_A, A_size);
    cudaMalloc((void **)&d_B, B_size);
    cudaMalloc((void **)&d_C, C_size);

    // Copy data to device
    cudaMemcpy(d_A, A.getDataConstPtr(), A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.getDataConstPtr(), B_size, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(1, 1, 1);

    int A_ncols = A.getCols();
    int A_nrows = A.getRows();
    int B_ncols = B.getCols();

    if (method == MultMethod::Standard)
    {
        // Fixed: Use proper dimensions and limits

        blocksPerGrid.x = (A_ncols + threadsPerBlock.x - 1) / threadsPerBlock.x;
        blocksPerGrid.y = (A_nrows + threadsPerBlock.y - 1) / threadsPerBlock.y;

        matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, A_nrows, A_ncols, B_ncols);
    }
    else if (method == MultMethod::Tiled)
    {
        blocksPerGrid.y = (A_nrows + threadsPerBlock.y - 1) / threadsPerBlock.y;

        matrixMulTiledKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, A_nrows, A_ncols, B_ncols);
    }
    else if (method == MultMethod::Granular)
    {
        blocksPerGrid.x = (A_ncols + threadsPerBlock.x - 1) / threadsPerBlock.x;

        matrixMulGranularKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, A_ncols, A_nrows, B_ncols);
    }
    else
    {
        // Clean up before throwing
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("Incorrect Method");
    }

    cudaDeviceSynchronize(); // Ensure kernel completes before copying

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Copy result back
    cudaMemcpy(C.getDataPtr(), d_C, C_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

Matrix multMatrixMatrixOnHost(const Matrix &A, const Matrix &B)
{
    if (A.getCols() != B.getRows())
    {
        throw std::runtime_error("Incompatible matrix dimensions for multiplication");
    }

    Matrix C(A.getRows(), B.getCols());
    for (unsigned int i = 0; i < A.getRows(); ++i)
    {
        for (unsigned int j = 0; j < B.getCols(); ++j)
        {
            for (unsigned int k = 0; k < A.getCols(); ++k)
            {
                C.getDataPtr()[i * C.getCols() + j] +=
                    A.getDataConstPtr()[i * A.getCols() + k] *
                    B.getDataConstPtr()[k * B.getCols() + j];
            }
        }
    }
    return C;
}
