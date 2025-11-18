#include "scan.h"

__global__ void kernelScan(int *out, const int *in, size_t n)
{
    __shared__ int block_mem[BLOCK_SIZE];
    uint threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int local_add = 0;

    if (threadId < n)
    {
        block_mem[threadIdx.x] = in[threadId];
    }
    else
    {
        block_mem[threadIdx.x] = 0;
    }

    __syncthreads();

    for (uint i = 0; (1<<i) < BLOCK_SIZE; i++)
    {
        local_add = 0;
        if (threadIdx.x>((1<<i)-1))
        {
            local_add = block_mem[(threadIdx.x-(1<<i))];
        }

        __syncthreads();

        if (threadIdx.x>((1<<i)-1))
        {
            atomicAdd(&block_mem[threadIdx.x], local_add);
        }
        
        __syncthreads();
    }

    out[threadId] = block_mem[threadIdx.x];
}

__global__ void kernelAddSums(int *out, const int *sums, size_t n)
{
    __shared__ int block_mem[BLOCK_SIZE];
    uint threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int prevSumIndx = blockIdx.x * blockDim.x - 1;

    if (threadId*BLOCK_SIZE < n)
    {
        block_mem[threadIdx.x] = sums[threadId*BLOCK_SIZE];
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

    out[threadId] = block_mem[threadIdx.x];
}

std::vector<int> scanOnDevice(const std::vector<int> &in, ScanMethod method)
{
    cudaError_t err = cudaSuccess;
    std::vector<int> output = {0};
    output.reserve(in.size());

    int *d_in = nullptr;
    int *d_out = nullptr;

    int  width = in.size();
    int size_width = width*sizeof(in[0]);

    err = cudaMalloc((void **)&d_in, size_width);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaMalloc((void **)&d_out, size_width);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    int blockSize = BLOCK_SIZE;
    int numBlocks = (width + blockSize - 1) / blockSize;

    cudaMemcpy(d_in, in.data(), size_width, cudaMemcpyHostToDevice);

    if (numBlocks < 1) numBlocks = 1;

    kernelScan<<<numBlocks, blockSize>>>(d_out, d_in, width);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {    
        cudaFree(d_in);
        cudaFree(d_out);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out); 
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Copy result back
    err = cudaMemcpy(output.data(), d_out, size_width, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;

}

std::vector<int> scanOnHost(const std::vector<int> &in)
{
    std::vector<int> out(in.size());
    if (in.size() == 0)
    {
        return out;
    }

    out[0] = in[0];
    for (size_t i = 1; i < in.size(); ++i)
    {
        out[i] = out[i - 1] + in[i];
    }

    return out;
}
