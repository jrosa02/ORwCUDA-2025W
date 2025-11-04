#include "conv_1d.h"
#include <stdexcept>

__constant__ static float c_mask[MAX_MASK_WIDTH];

__global__ void conv1dBasicKernel(float *output, const float *signal, const int width, const int maskWidth)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int position = 0;
    if (threadId >= width) return;

    float signal_probe = 0;
    float accumulator = 0;

    for (size_t i = 0; i < maskWidth; ++i)
    {
        position = threadId-maskWidth+((int)i);
        if ( position > 0 && position < width)
        {
            signal_probe = signal[position];
        }
        else
        {
            signal_probe = 0;
        }

        accumulator += signal_probe*c_mask[maskWidth-i];
    }
    output[threadId] = accumulator;
}

__global__ void conv1dTiledKernel(float *output, const float *signal, const int width, const int maskWidth)
{
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

    cudaMemcpyToSymbol(c_mask, mask.data(), mask.size()*sizeof(mask[0]));

    if (numBlocks < 1) numBlocks = 1;
    // Launch appropriate kernel
    if (method == ConvMethod::Basic)
    {
        conv1dBasicKernel<<<numBlocks, blockSize>>>(d_output, d_signal, width, mask.size());
    }
    else if (method == ConvMethod::Tiled)
    {
        // conv1dTiledKernel<<<numBlocks, blockSize>>>();
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
