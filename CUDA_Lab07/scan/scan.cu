#include "scan.h"
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(x) do { if ((x) != cudaSuccess) \
    throw std::runtime_error(cudaGetErrorString(x)); } while(0)

// Kernel 1: skan w obrębie bloków + zapis sum blokowych
__global__ void kernelBlockScanInclusive(const int *in, int *out, int n, int *blockSums)
{
    __shared__ int sdata[BLOCK_SIZE];
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tid = threadIdx.x;

    sdata[tid] = (gid < n ? in[gid] : 0);
    __syncthreads();

    for (unsigned offset = 1; offset < blockDim.x; offset <<= 1) {
        int val = (tid >= offset ? sdata[tid - offset] : 0);
        __syncthreads();
        sdata[tid] += val;
        __syncthreads();
    }

    if (gid < n) out[gid] = sdata[tid];
    if (tid == blockDim.x - 1) blockSums[blockIdx.x] = sdata[tid];
}

// Kernel 2: dodanie przesunięć blokowych
__global__ void kernelAddBlockOffsets(int *out, const int *offsets, int n)
{
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n || blockIdx.x == 0) return;
    out[gid] += offsets[blockIdx.x - 1];
}

// CPU scan
static void cpuScan(std::vector<int> &v)
{
    for (size_t i = 1; i < v.size(); ++i) v[i] += v[i - 1];
}

// Główna funkcja
std::vector<int> scanOnDevice(const std::vector<int> &in, ScanMethod method)
{
    if (in.empty()) return {};
    int n = in.size();

    int *d_in, *d_out, *d_blockSums;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMalloc(&d_blockSums, numBlocks * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    kernelBlockScanInclusive<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, n, d_blockSums);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (numBlocks > 1) {
        std::vector<int> blockSums(numBlocks);
        CUDA_CHECK(cudaMemcpy(blockSums.data(), d_blockSums, numBlocks*sizeof(int), cudaMemcpyDeviceToHost));
        cpuScan(blockSums);
        CUDA_CHECK(cudaMemcpy(d_blockSums, blockSums.data(), numBlocks*sizeof(int), cudaMemcpyHostToDevice));
        kernelAddBlockOffsets<<<numBlocks, BLOCK_SIZE>>>(d_out, d_blockSums, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<int> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_blockSums);
    return out;
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
