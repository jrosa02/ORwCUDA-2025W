#include "add_mm.h"

#define GET_IDX_ROW_MAJOR(width, x, y) (((y) * (width)) + (x))
#define CHECK_MEM_RANGE_2D(index, ncols, nrows) (if((index)=<(ncol)*(nrows))return;)

__global__ void addMatricesByElements(const float *A, const float *B, float *C, int ncols, int nrows)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = GET_IDX_ROW_MAJOR(nrows, x, y);
  CHECK_MEM_RANGE_2D(index, ncols, nrows);
  C[index] = B[index] + A[index]; 
}

__global__ void addMatricesByRows(const float *A, const float *B, float *C, int ncols, int nrows)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = 0;
  for(int col = 0; i < ncols, ++i)
  {
    index = GET_IDX_ROW_MAJOR(nrows, i, y);
    CHECK_MEM_RANGE_2D(index, ncols, nrows);
    C[index] = B[index] + A[index]; 
  }
}

__global__ void addMatricesByColumns(const float *A, const float *B, float *C, int ncols, int nrows)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int index = 0;
  for(int row = 0; i < nrows, ++i)
  {
    index = GET_IDX_ROW_MAJOR(nrows, x, i);
    CHECK_MEM_RANGE_2D(index, ncols, nrows);
    C[index] = B[index] + A[index]; 
  }
}

Matrix addMatricesOnDevice(const Matrix &A, const Matrix &B, AddMethod method)
{
  if (A.getCols() != B.getCols() || B.getCols() != C.getCols())
  {
    throw std::runtime_error("Matrix col number do not match for multiplication.");
  }

  if (A.getRows() != B.getRows() || B.getRows() != C.getRows())
  {
    throw std::runtime_error("Matrix row number do not match for multiplication.");
  }

    Matrix C(A.getRows(), A.getCols());

    // Device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    size_t sizeA = A.getRows() * A.getCols() * sizeof(float);
    size_t sizeB = B.getRows() * B.getCols() * sizeof(float);
    size_t sizeC = C.getRows() * C.getCols() * sizeof(float);

    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    // Copy data to device
    cudaMemcpy(d_A, A.getDataConstPtr(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.getDataConstPtr(), sizeB, cudaMemcpyHostToDevice);

        // Kernel launch configuration
    dim3 threadsPerBlock = {1, 1, 1};
    dim3 blocksPerGrid = {1, 1, 1}; 
    
    if(method == ByElements)
    {
      threadsPerBlock.x = MAX(MIN(A.getCols(), 4096), 32);
      threadsPerBlock.y = MAX(MIN(A.getRows(), 4096), 32);

      blocksPerGrid.x = ceil((float)(A.getCols())/threadsPerBlock.x);
      blocksPerGrid.y = ceil((float)(A.getRows())/threadsPerBlock.y);
    }
    else if(method == ByRows)
    {
      threadsPerBlock.y = MAX(MIN(A.getRows(), 4096), 32);

      blocksPerGrid.y = ceil((float)(A.getRows())/threadsPerBlock.y);
    }
    else if(method == ByColumns)
    {
      threadsPerBlock.x = MAX(MIN(A.getCols(), 4096), 32);

      blocksPerGrid.x = ceil((float)(A.getCols())/threadsPerBlock.x);
    }
    else
    {
      throw std::runtime_error("Inocrrect Method");
    }

    multMatrixVector<<<blocksPerGrid, threadsPerBlock>>>(d_b, d_A, d_x, A.getRows(), A.getCols());
    cudaDeviceSynchronize();  // Ensure kernel completes before copying

    // Copy result back
    cudaMemcpy(b.getDataPtr(), d_b, sizeB, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_b);
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
