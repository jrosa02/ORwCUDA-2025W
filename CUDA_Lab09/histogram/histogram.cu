#include "histogram.h"

#define PART_SIZE (0xFF)
#define NUMBER_OF_STREAMS (4)

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
    __shared__ unsigned int localHist[256]; // shared memory dla histogramu bloku
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Zainicjalizuj shared memory
    for (int i = tid; i < nBins; i += blockDim.x)
        localHist[i] = 0;
    __syncthreads();

    // Obliczanie histogramu lokalnego
    if (idx < size)
    {
        unsigned char val = buffer[idx];
        int bin = val / (256 / nBins);
        atomicAdd(&localHist[bin], 1);
    }
    __syncthreads();

    // Sumowanie do globalnego histogramu
    for (int i = tid; i < nBins; i += blockDim.x)
    {
        atomicAdd(&histogram[i], localHist[i]);
    }
}


std::vector<unsigned int> computeHistogramOnDevice(const std::vector<unsigned char> &data, int nBins, HistMethod method)
{
    std::vector<unsigned int> finalHist(nBins, 0);

    unsigned char *h_pinnedData = nullptr;
    unsigned int *h_pinnedHist = nullptr;

    size_t dataSize = data.size() * sizeof(unsigned char);
    size_t histSize = nBins * NUMBER_OF_STREAMS * sizeof(unsigned int);

    cudaMallocHost(&h_pinnedData, dataSize);
    cudaMallocHost(&h_pinnedHist, histSize);
    memcpy(h_pinnedData, data.data(), dataSize);
    memset(h_pinnedHist, 0, histSize);

    cudaStream_t streams[NUMBER_OF_STREAMS];
    for (int i = 0; i < NUMBER_OF_STREAMS; ++i)
        cudaStreamCreate(&streams[i]);

    size_t streamSize = data.size() / NUMBER_OF_STREAMS;
    unsigned char *d_data;
    unsigned int *d_hist;

    cudaMalloc(&d_data, streamSize * sizeof(unsigned char));
    cudaMalloc(&d_hist, nBins * NUMBER_OF_STREAMS * sizeof(unsigned int));
    cudaMemset(d_hist, 0, nBins * NUMBER_OF_STREAMS * sizeof(unsigned int));

    for (int i = 0; i < NUMBER_OF_STREAMS; ++i) {
        size_t offset = i * streamSize;
        size_t currentSize = (i == NUMBER_OF_STREAMS - 1) ? (data.size() - offset) : streamSize;

        cudaMemcpyAsync(d_data, h_pinnedData + offset, currentSize * sizeof(unsigned char),
                        cudaMemcpyHostToDevice, streams[i]);

        int threads = 512;
        int blocks = 256;
        histogram_3<<<blocks, threads, 0, streams[i]>>>(d_data, currentSize, d_hist + i * nBins, nBins);

        cudaMemcpyAsync(h_pinnedHist + i * nBins, d_hist + i * nBins,
                        nBins * sizeof(unsigned int), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUMBER_OF_STREAMS; ++i)
        cudaStreamSynchronize(streams[i]);

    // Połączenie histogramów częściowych
    for (int i = 0; i < NUMBER_OF_STREAMS; ++i)
        for (int j = 0; j < nBins; ++j)
            finalHist[j] += h_pinnedHist[i * nBins + j];

    // Czyszczenie
    for (int i = 0; i < NUMBER_OF_STREAMS; ++i)
        cudaStreamDestroy(streams[i]);
    cudaFree(d_data);
    cudaFree(d_hist);
    cudaFreeHost(h_pinnedData);
    cudaFreeHost(h_pinnedHist);

    return finalHist;
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
