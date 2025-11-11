#include "conv_1d.h"
#include <stdexcept>

__constant__ static float c_mask[MAX_MASK_WIDTH];

__global__ void conv1dBasicKernel(float *output, const float *signal, const int width, const int maskWidth)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int position = 0;
    if (threadId >= width) return;

    float accumulator = 0;

    for (size_t i = 0; i < maskWidth; ++i)
    {
        position = threadId-maskWidth/2+((int)i);
        if ( position >= 0 && position < width)
        {
          accumulator +=  signal[position]*c_mask[i];;
        }
    }
    output[threadId] = accumulator;
}

__global__ void conv1dTiledKernel(float *output, const float *signal, int width, int maskWidth)
{
   extern __shared__ float shared_signal[];

    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int localId  = threadIdx.x;

    int radius = maskWidth / 2;
    int sharedSize = blockDim.x + maskWidth - 1;

    int start = blockIdx.x * blockDim.x - radius;

    for (int i = localId; i < sharedSize; i += blockDim.x)
    {
        int globalIndex = start + i;
        if (globalIndex >= 0 && globalIndex < width)
            shared_signal[i] = signal[globalIndex];
        else
            shared_signal[i] = 0.0f; // zero padding
    }

    __syncthreads();

    if (globalId < width)
    {
        float accumulator = 0.0f;
        for (int j = 0; j < maskWidth; ++j)
        {
            accumulator += shared_signal[localId + j] * c_mask[j];
        }
        output[globalId] = accumulator;
    }
}


std::vector<float> convolutionOnDevice(const std::vector<float> &signal, const std::vector<float> &mask, ConvMethod method)
{
    cudaError_t err = cudaSuccess;

    float *d_signal = nullptr;
    float *d_output = nullptr;

    int  width = signal.size();
    int size_width = width*sizeof(signal[0]);
    std::vector<float> output(width, 0.0f);

    err = cudaMalloc((void **)&d_signal, size_width);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaMalloc((void **)&d_output, size_width);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    int blockSize = 256;
    int numBlocks = (width + blockSize - 1) / blockSize;
    int sharedMemSize = (blockSize + mask.size() - 1) * sizeof(float);

    cudaMemcpyToSymbol(c_mask, mask.data(), mask.size()*sizeof(mask[0]));
    cudaMemcpy(d_signal, signal.data(), size_width, cudaMemcpyHostToDevice);

    if (numBlocks < 1) numBlocks = 1;

    if (method == ConvMethod::Basic)
    {
        conv1dBasicKernel<<<numBlocks, blockSize>>>(d_output, d_signal, width, mask.size());
    }
    else if (method == ConvMethod::Tiled)
    {
        conv1dTiledKernel<<<numBlocks, blockSize, sharedMemSize>>>(d_output, d_signal, width, mask.size());
    }
    else
    {
        throw std::runtime_error("Wrong Method selected");
    }
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {    
        cudaFree(d_signal);
        cudaFree(d_output);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_signal);
        cudaFree(d_output); 
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Copy result back
    err = cudaMemcpy(output.data(), d_output, size_width, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_signal);
        cudaFree(d_output);
        throw std::runtime_error(cudaGetErrorString(err));
    }


    return output;
}

std::vector<float> convolutionOnHost(const std::vector<float> &signal, const std::vector<float> &mask)
{
    int signalWidth = static_cast<int>(signal.size());
    int maskWidth = static_cast<int>(mask.size());
    int outputWidth = signalWidth;

    std::vector<float> output(outputWidth, 0.0f);

    // Convolution with zero padding
    int n = maskWidth / 2;
    for (int idxP = 0; idxP < outputWidth; ++idxP)
    {
        float convAccum = 0.0f;
        for (int i = idxP - n; i <= idxP + n; ++i)
        {
            if (i >= 0 && i < signalWidth)
            {
                convAccum += signal[i] * mask[i - (idxP - n)];
            }
        }
        output[idxP] = convAccum;
    }

    return output;
}
