#include "histogram.h"

#define PART_SIZE (0xFF)

// Histogram - basic parallel implementation
__global__ void histogram_1(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
    int indx = blockIdx.x * blockDim.x + threadIdx.x;
    uint txt_indx = 0u;
    uint hist_indx = 0u;
    unsigned char letter = 0u;

    // Check bounds to avoid illegal memory access
    #pragma unroll
    for (size_t i = 0; i < PART_SIZE; i++)
    {
        txt_indx = indx*PART_SIZE + i;
        if (indx < size)
        {
            letter = buffer[txt_indx];
            hist_indx = letter/(255/nBins);
            atomicAdd(&histogram[hist_indx], 1);
        }
    }
}

// Histogram - interleaved partitioning
__global__ void histogram_2(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
}

// Histogram - interleaved partitioning + privatisation
__global__ void histogram_3(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
}


std::vector<unsigned int> computeHistogramOnDevice(const std::vector<unsigned char> &data, int nBins, HistMethod method)
{

    std::vector<unsigned int> histogram;
    histogram.reserve(data.size());

    unsigned char *d_data = nullptr;
    uint *d_histogram = nullptr;

    size_t data_size = data.size() * sizeof(unsigned char);
    size_t histogram_size = histogram.size() * sizeof(unsigned char);

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
    
    // Copy data to device
    err = cudaMemcpy(d_data, data.data(), data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_histogram);
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Kernel launch configuration - moved after error checking
    dim3 threadsPerBlock(32);
    dim3 blocksPerGrid(
        ceil(data.size() / threadsPerBlock.x)
    );

    // Launch appropriate kernel
    if (method == HistMethod::Block)
    {
        histogram_1<<<blocksPerGrid, threadsPerBlock>>>(d_data, data_size, d_histogram, nBins);
    }
    else if (method == HistMethod::Interleaved)
    {
        histogram_2<<<blocksPerGrid, threadsPerBlock>>>(d_data, data_size, d_histogram, nBins);
    }
    else if (method == HistMethod::Privatised)
    {
        histogram_3<<<blocksPerGrid, threadsPerBlock>>>(d_data, data_size, d_histogram, nBins);
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

    cudaDeviceSynchronize();

    // Copy result back
    err = cudaMemcpy(histogram.data(), d_histogram, nBins, cudaMemcpyDeviceToHost);
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
