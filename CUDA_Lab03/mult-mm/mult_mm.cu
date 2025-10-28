#include "mult_mm.h"

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
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float accumulator = 0.0f;
    
    for (int k = 0; k < (A_cols + TILE_SIZE - 1) / TILE_SIZE; ++k)
    {
        // Load tiles from global to shared memory
        int a_col = k * TILE_SIZE + threadIdx.x;
        int b_row = k * TILE_SIZE + threadIdx.y;
        
        if (row < A_rows && a_col < A_cols)
            tileA[threadIdx.y][threadIdx.x] = A[row * A_cols + a_col];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (b_row < A_cols && col < B_cols)
            tileB[threadIdx.y][threadIdx.x] = B[b_row * B_cols + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Compute partial product
        for (int i = 0; i < TILE_SIZE; ++i)
        {
            accumulator += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < A_rows && col < B_cols)
    {
        C[row * B_cols + col] = accumulator;
    }   
}

__global__ void matrixMulGranularKernel(const float *A, const float *B, float *C,
                                        int A_rows, int A_cols, int B_cols)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    // Fine-grained version - each thread computes multiple elements
    const int ELEMENTS_PER_THREAD = 4;
    
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i)
    {
        int row = blockIdx.y * blockDim.y * ELEMENTS_PER_THREAD + threadIdx.y * ELEMENTS_PER_THREAD + i;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        float accumulator = 0.0f;

        for (int k = 0; k < (A_cols + TILE_SIZE - 1) / TILE_SIZE; ++k)
            {
                // Load tiles from global to shared memory
                int a_col = k * TILE_SIZE + threadIdx.x;
                int b_row = k * TILE_SIZE + threadIdx.y;
                
                if (row < A_rows && a_col < A_cols)
                    tileA[threadIdx.y][threadIdx.x] = A[row * A_cols + a_col];
                else
                    tileA[threadIdx.y][threadIdx.x] = 0.0f;
                    
                if (b_row < A_cols && col < B_cols)
                    tileB[threadIdx.y][threadIdx.x] = B[b_row * B_cols + col];
                else
                    tileB[threadIdx.y][threadIdx.x] = 0.0f;
                
                __syncthreads();
                
                // Compute partial product
                for (int i = 0; i < TILE_SIZE; ++i)
                {
                    accumulator += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
                }
                
                __syncthreads();
            }
            
            if (row < A_rows && col < B_cols)
            {
                C[row * B_cols + col] = accumulator;
            }   
    }
}

Matrix multMatrixMatrixOnDevice(const Matrix &A, const Matrix &B, MultMethod method)
{
    if (A.getCols() != B.getRows())
    {
        throw std::runtime_error("Matrix dimensions do not match for multiplication.");
    }

    Matrix C(A.getRows(), B.getCols());

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    size_t A_size = A.getRows() * A.getCols() * sizeof(float);
    size_t B_size = B.getRows() * B.getCols() * sizeof(float);
    size_t C_size = C.getRows() * C.getCols() * sizeof(float);

    cudaError_t err = cudaSuccess;
    
    // Allocate memory with error checking
    err = cudaMalloc((void **)&d_A, A_size);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    err = cudaMalloc((void **)&d_B, B_size);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    err = cudaMalloc((void **)&d_C, C_size);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Copy data to device
    err = cudaMemcpy(d_A, A.getDataConstPtr(), A_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    err = cudaMemcpy(d_B, B.getDataConstPtr(), B_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Kernel launch configuration - moved after error checking
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (B.getCols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (A.getRows() + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    int A_rows = A.getRows();
    int A_cols = A.getCols();
    int B_cols = B.getCols();

    // Launch appropriate kernel
    if (method == MultMethod::Standard)
    {
        matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, A_rows, A_cols, B_cols);
    }
    else if (method == MultMethod::Tiled)
    {
        matrixMulTiledKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, A_rows, A_cols, B_cols);
    }
    else if (method == MultMethod::Granular)
    {
        matrixMulGranularKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, A_rows, A_cols, B_cols);
    }
    else
    {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("Incorrect Method");
    }

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    // Copy result back
    err = cudaMemcpy(C.getDataPtr(), d_C, C_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error(cudaGetErrorString(err));
    }

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
