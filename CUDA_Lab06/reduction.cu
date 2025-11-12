#include "reduction.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <stdio.h>

namespace cg = cooperative_groups;

__global__ void reductionKernelBasic(int *sum, int *input, int width)
{
    __shared__ int block_mem[BLOCK_SIZE];
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId>width) return;

    for (size_t i = 0; i < BLOCK_SIZE; i++)
    {
        block_mem[i] = input[blockIdx.x*BLOCK_SIZE+i];
    }

    for (uint i = 1; (1<<(i-1)) < width; i++)
    {
        if (threadId%(1<<i)==0)
        {
            atomicAdd(&input[threadId], input[threadId+(1<<i-1)]);
        }
        __syncthreads();
    }
    if (threadId == 0)
    {
        *sum = input[0]; 
    }
}

__global__ void reductionKernelOptimized(int *sum, int *input, int width)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= width) return;
}

__global__ void reductionKernelCooperativeGroups(int *sum, const int *input, int width)
{
}


int reductionOnDevice(const std::vector<int> &data, ReductionMethod method)
{
    cudaError_t err = cudaSuccess;

    int *d_data = nullptr;
    int *d_output = nullptr;

    int  width = data.size();
    int size_width = width*sizeof(data[0]);
    int output = 0;

    err = cudaMalloc((void **)&d_data, size_width);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaMalloc((void **)&d_output, sizeof(int));
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    int blockSize = BLOCK_SIZE;
    int numBlocks = (width + blockSize - 1) / blockSize;
    int sharedMemSize = (blockSize - 1) * sizeof(output);

    cudaMemcpy(d_data, data.data(), size_width, cudaMemcpyHostToDevice);

    if (numBlocks < 1) numBlocks = 1;

    if (method == ReductionMethod::Basic)
    {
        reductionKernelBasic<<<numBlocks, blockSize>>>(d_output, d_data, width);
    }
    else if (method == ReductionMethod::Optimized)
    {
        reductionKernelOptimized<<<numBlocks, blockSize, sharedMemSize>>>(d_output, d_data, width);
    }
    else if (method == ReductionMethod::CooperativeGroups)
    {
        reductionKernelCooperativeGroups<<<numBlocks, blockSize, sharedMemSize>>>(d_output, d_data, width);
    }
    
    else
    {
        throw std::runtime_error("Wrong Method selected");
    }
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {    
        cudaFree(d_data);
        cudaFree(d_output);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_output); 
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Copy result back
    err = cudaMemcpy(&output, d_output, sizeof(output), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_output);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

int reductionOnHost(const std::vector<int> &data)
{
    int sum = 0;
    for (const auto &val : data)
    {
        sum += val;
    }
    return sum;
}
