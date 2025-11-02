#include "histogram.h"


#define PART_SIZE (4)  // Reduced for better load balancing
#define N_LETTERS 26

// Histogram - basic parallel implementation
__global__ void histogram_1(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;
    
    int binWidth = (N_LETTERS + nBins - 1) / nBins; // ceiling division - MUST match host
    
    // Each thread processes a contiguous block of elements
    long chunkSize = (size + totalThreads - 1) / totalThreads;
    long start = threadId * chunkSize;
    long end = min(start + chunkSize, size);
    
    for (long i = start; i < end; i++)
    {
        unsigned char ch = buffer[i];
        // MUST match host logic exactly
        if (ch >= 'a' && ch <= 'z') {
            int alphabetPosition = ch - 'a';
            int binIndex = alphabetPosition / binWidth;
            if (binIndex < nBins) {
                atomicAdd(&histogram[binIndex], 1);
            }
        }
    }
}

// Histogram - interleaved partitioning
__global__ void histogram_2(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    
    int binWidth = (N_LETTERS + nBins - 1) / nBins; // ceiling division - MUST match host
    
    // Each thread processes elements with stride = totalThreads
    for (long i = tid; i < size; i += totalThreads)
    {
        unsigned char ch = buffer[i];
        // MUST match host logic exactly
        if (ch >= 'a' && ch <= 'z') {
            int alphabetPosition = ch - 'a';
            int binIndex = alphabetPosition / binWidth;
            if (binIndex < nBins) {
                atomicAdd(&histogram[binIndex], 1);
            }
        }
    }
}

// Histogram - interleaved partitioning + privatisation
__global__ void histogram_3(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
    // Shared memory for thread block privatization
    extern __shared__ unsigned int shared_hist[];
    
    int tid = threadIdx.x;
    int blockId = blockIdx.x;
    int blockDimX = blockDim.x;
    
    int binWidth = (N_LETTERS + nBins - 1) / nBins; // ceiling division - MUST match host
    
    // Initialize shared memory histogram
    for (int i = tid; i < nBins; i += blockDimX)
    {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Process data with interleaved partitioning
    int totalThreads = blockDimX * gridDim.x;
    long startIdx = blockId * blockDimX + tid;
    
    for (long i = startIdx; i < size; i += totalThreads)
    {
        unsigned char ch = buffer[i];
        // MUST match host logic exactly
        if (ch >= 'a' && ch <= 'z') {
            int alphabetPosition = ch - 'a';
            int binIndex = alphabetPosition / binWidth;
            if (binIndex < nBins) {
                atomicAdd(&shared_hist[binIndex], 1);
            }
        }
    }
    __syncthreads();
    
    // Merge local histograms to global memory
    for (int i = tid; i < nBins; i += blockDimX)
    {
        if (shared_hist[i] > 0)
        {
            atomicAdd(&histogram[i], shared_hist[i]);
        }
    }
}

std::vector<unsigned int> computeHistogramOnDevice(const std::vector<unsigned char> &data, int nBins, HistMethod method)
{
    if (nBins < 1 || nBins > 26) {
        throw std::runtime_error("Number of bins must be between 1 and 26");
    }

    std::vector<unsigned int> histogram(nBins, 0);  // Initialize with zeros

    unsigned char *d_data = nullptr;
    unsigned int *d_histogram = nullptr;

    size_t data_size = data.size() * sizeof(unsigned char);
    size_t histogram_size = nBins * sizeof(unsigned int);

    cudaError_t err = cudaSuccess;
    
    // Allocate memory with error checking
    err = cudaMalloc((void **)&d_data, data_size);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    err = cudaMalloc((void **)&d_histogram, histogram_size);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Initialize histogram to zeros on device
    err = cudaMemset(d_histogram, 0, histogram_size);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_histogram);
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Copy data to device
    err = cudaMemcpy(d_data, data.data(), data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_histogram);
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Kernel launch configuration
    int blockSize = 256;
    int numBlocks = (data.size() + blockSize - 1) / blockSize;
    
    // Adjust for PART_SIZE in histogram_1 - ensure we don't create too many blocks
    if (method == HistMethod::Block) {
        numBlocks = (data.size() + blockSize * PART_SIZE - 1) / (blockSize * PART_SIZE);
        numBlocks = min(numBlocks, 60); // Limit as per instructions
    } else {
        numBlocks = min(numBlocks, 60); // Limit as per instructions
    }

    // Ensure at least 1 block
    if (numBlocks < 1) numBlocks = 1;

    // Launch appropriate kernel
    if (method == HistMethod::Block)
    {
        histogram_1<<<numBlocks, blockSize>>>(d_data, data.size(), d_histogram, nBins);
    }
    else if (method == HistMethod::Interleaved)
    {
        histogram_2<<<numBlocks, blockSize>>>(d_data, data.size(), d_histogram, nBins);
    }
    else if (method == HistMethod::Privatised)
    {
        // Calculate shared memory size for privatization
        size_t sharedMemSize = nBins * sizeof(unsigned int);
        histogram_3<<<numBlocks, blockSize, sharedMemSize>>>(d_data, data.size(), d_histogram, nBins);
    }
    else
    {
        cudaFree(d_data);
        cudaFree(d_histogram);
        throw std::runtime_error("Incorrect Method");
    }

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_histogram);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_histogram);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Copy result back
    err = cudaMemcpy(histogram.data(), d_histogram, histogram_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_histogram);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_histogram);

    return histogram;
}


std::vector<unsigned int> computeHistogramOnHost(const std::vector<unsigned char> &data, int nBins)
{
    std::vector<unsigned int> histogram(nBins, 0);
    int binWidth = (N_LETTERS + nBins - 1) / nBins; // ceiling division

    for (const auto &ch : data)
    {
        int alphabetPosition = ch - 'a';
        if (alphabetPosition >= 0 && alphabetPosition < N_LETTERS)
        {
            histogram[alphabetPosition / binWidth]++;
        }
    }

    return histogram;
}