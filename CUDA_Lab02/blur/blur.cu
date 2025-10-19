#include "blur.h"

#define TILE_WIDTH  16
#define CHANNELS    1
#define BLUR_SIZE   3

#define GET_IDX_ROW_MAJOR(width, x, y) (((y) * (width)) + (x))

__global__ void blurKernel(float *out, float *in, int width, int height, int blurSize)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (!(x < width && y < height))
    {
        return;
    }

    // get 1D coordinate for the grayscale image
    const int center_pixel_index = GET_IDX_ROW_MAJOR(width, x, y);
    int pixel_index = 0;
    float pixel_val = 0;
    float out_pixel_acc = 0;

    //#pragma unroll
    for (int j = -blurSize/2; j <= blurSize/2; j++)
    {
        //#pragma unroll
        for (int i = -blurSize/2; i <= blurSize/2; i++)
        {
            if ((x+i < 0 || x+i >= width || y+j < 0 || y+j >= height))
            {
                pixel_val = 0.0;
            }
            else
            {
                pixel_index = GET_IDX_ROW_MAJOR(width, x+i, y+j);
                pixel_val = in[pixel_index];
            }

            out_pixel_acc += pixel_val;
        }
    }

    out[center_pixel_index] = out_pixel_acc/float((blurSize*blurSize));

    return;
}

Image imageBlurOnDevice(const Image &inputImage, int blurSize)
{
    Image outputImage(inputImage.getWidth(), inputImage.getHeight(), inputImage.isGray());

    int contextRadius = (blurSize - 1) / 2;

    // allocate input and output images in the device
    float *d_inputImage;
    float *d_outputImage;
    cudaMalloc((void **)&d_inputImage, inputImage.getRows() * inputImage.getCols() * sizeof(float));
    cudaMalloc((void **)&d_outputImage, outputImage.getRows() * outputImage.getCols() * sizeof(float));

    // copy image to the device
    cudaMemcpy(d_inputImage, inputImage.getDataConstPtr(), inputImage.getRows() * inputImage.getCols() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil((float)outputImage.getCols() / TILE_WIDTH), ceil((float)outputImage.getRows() / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    blurKernel<<<dimGrid, dimBlock>>>(d_outputImage, d_inputImage, outputImage.getCols(), outputImage.getRows(), blurSize);

    cudaMemcpy(outputImage.getDataPtr(), d_outputImage, outputImage.getRows() * outputImage.getCols() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return outputImage;
}

Image imageBlurOnHost(const Image &inputImage, int blurSize)
{
    Image outputImage(inputImage.getWidth(), inputImage.getHeight(), inputImage.isGray());

    int contextRadius = (blurSize - 1) / 2;

    for (unsigned int y = 0; y < inputImage.getHeight(); ++y)
    {
        for (unsigned int x = 0; x < inputImage.getWidth(); ++x)
        {
            float outVal = 0.0f;
            // Inside full context space
            if (x >= contextRadius && x < inputImage.getWidth() - contextRadius && y >= contextRadius && y < inputImage.getHeight() - contextRadius)
            {
                float accumVal = 0.0f;
                for (int c = -contextRadius; c <= contextRadius; c++)
                {
                    for (int r = -contextRadius; r <= contextRadius; ++r)
                    {
                        int accumIdx = (y + c) * inputImage.getWidth() + (x + r);
                        accumVal += inputImage.getDataConstPtr()[accumIdx];
                    }
                }
                outVal = accumVal / (blurSize * blurSize);
            }

            int outIdx = y * inputImage.getWidth() + x;
            outputImage.getDataPtr()[outIdx] = outVal;
        }
    }

    return outputImage;
}
