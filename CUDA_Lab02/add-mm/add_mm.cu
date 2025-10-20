#include "add_mm.h"

#define GET_IDX_ROW_MAJOR(width, x, y) (((y) * (width)) + (x))
#define CHECK_MEM_RANGE_2D(index, ncols, nrows) if((index) >= ((ncols) * (nrows))) return;

__global__ void addMatricesByElements(const float *A, const float *B, float *C, int ncols, int nrows)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  // Check bounds before accessing memory
  if (x >= ncols || y >= nrows) return;
  
  int index = GET_IDX_ROW_MAJOR(ncols, x, y);  // Fixed: should be ncols, not nrows
  C[index] = A[index] + B[index];  // Fixed: A + B, not B + A (more conventional)
}

__global__ void addMatricesByRows(const float *A, const float *B, float *C, int ncols, int nrows)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  // Check row bounds
  if (y >= nrows) return;
  
  for(int col = 0; col < ncols; ++col)  // Fixed: variable name and condition
  {
    int index = GET_IDX_ROW_MAJOR(ncols, col, y);  // Fixed: should be ncols
    C[index] = A[index] + B[index];
  }
}

__global__ void addMatricesByColumns(const float *A, const float *B, float *C, int ncols, int nrows)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Check column bounds
  if (x >= ncols) return;
  
  for(int row = 0; row < nrows; ++row)  // Fixed: variable name and condition
  {
    int index = GET_IDX_ROW_MAJOR(ncols, x, row);  // Fixed: should be ncols
    C[index] = A[index] + B[index];
  }
}

Matrix addMatricesOnDevice(const Matrix &A, const Matrix &B, AddMethod method)
{
  // Fixed: Check only A and B dimensions
  if (A.getCols() != B.getCols() || A.getRows() != B.getRows())
  {
    throw std::runtime_error("Matrix dimensions do not match for addition.");
  }

  Matrix C(A.getRows(), A.getCols());

  // Device memory
  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  size_t size = A.getRows() * A.getCols() * sizeof(float);

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  // Copy data to device
  cudaMemcpy(d_A, A.getDataConstPtr(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.getDataConstPtr(), size, cudaMemcpyHostToDevice);

  // Kernel launch configuration
  dim3 threadsPerBlock(1, 1, 1);
  dim3 blocksPerGrid(1, 1, 1); 
  
  int ncols = A.getCols();
  int nrows = A.getRows();
  
  if(method == ByElements)
  {
    // Fixed: Use proper dimensions and limits
    threadsPerBlock.x = min(max(ncols, 1), 32);
    threadsPerBlock.y = min(max(nrows, 1), 32);

    blocksPerGrid.x = (ncols + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocksPerGrid.y = (nrows + threadsPerBlock.y - 1) / threadsPerBlock.y;
    
    addMatricesByElements<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, ncols, nrows);
  }
  else if(method == ByRows)
  {
    threadsPerBlock.y = min(max(nrows, 1), 256);  // Can use larger blocks for 1D
    blocksPerGrid.y = (nrows + threadsPerBlock.y - 1) / threadsPerBlock.y;
    
    addMatricesByRows<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, ncols, nrows);
  }
  else if(method == ByColumns)
  {
    threadsPerBlock.x = min(max(ncols, 1), 256);  // Can use larger blocks for 1D
    blocksPerGrid.x = (ncols + threadsPerBlock.x - 1) / threadsPerBlock.x;
    
    addMatricesByColumns<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, ncols, nrows);
  }
  else
  {
    // Clean up before throwing
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    throw std::runtime_error("Incorrect Method");
  }

  cudaDeviceSynchronize();  // Ensure kernel completes before copying

  // Check for kernel errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    throw std::runtime_error(cudaGetErrorString(err));
  }

  // Copy result back
  cudaMemcpy(C.getDataPtr(), d_C, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  return C;
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