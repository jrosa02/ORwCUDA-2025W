#include "reduction.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <stdio.h>

namespace cg = cooperative_groups;

__global__ void reductionKernelBasic(int *sum, int *input, int width)
{
    __shared__ int block_mem[BLOCK_SIZE];
    uint threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId>width) return;

    if (threadId < width)
    {
        block_mem[threadIdx.x] = input[threadId];
    }
    else
    {
        block_mem[threadIdx.x] = 0;
    }

    __syncthreads();

    for (uint i = 1; (1<<(i-1)) < BLOCK_SIZE; i++)
    {
        if (threadIdx.x%(1<<i)==0)
        {
            atomicAdd(&block_mem[threadIdx.x], block_mem[threadIdx.x+(1<<i-1)]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(sum, block_mem[0]); 
    }
}

__global__ void reductionKernelOptimized(int *sum, int *input, int width)
{
    __shared__ int block_mem[BLOCK_SIZE];
    uint threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId>width) return;

    if (threadId < width)
    {
        block_mem[threadIdx.x] = input[threadId];
    }
    else
    {
        block_mem[threadIdx.x] = 0;
    }

    __syncthreads();

    for (int i = 7; i >= 0; i--)
    {
        if (threadIdx.x < (1<<i))
        {
            atomicAdd(&block_mem[threadIdx.x], block_mem[threadIdx.x+(1<<i)]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(sum, block_mem[0]); 
    }
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

    cudaMemcpy(d_data, data.data(), size_width, cudaMemcpyHostToDevice);

    if (numBlocks < 1) numBlocks = 1;

    if (method == ReductionMethod::Basic)
    {
        reductionKernelBasic<<<numBlocks, blockSize>>>(d_output, d_data, width);
    }
    else if (method == ReductionMethod::Optimized)
    {
        reductionKernelOptimized<<<numBlocks, blockSize>>>(d_output, d_data, width);
    }
    else if (method == ReductionMethod::CooperativeGroups)
    {
        reductionKernelCooperativeGroups<<<numBlocks, blockSize>>>(d_output, d_data, width);
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
