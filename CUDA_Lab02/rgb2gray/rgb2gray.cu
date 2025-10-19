#include <image.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

#define TILE_WIDTH 16
__global__ void rgb2gray(float *grayImage, float *rgbImage, int channels, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        // get 1D coordinate for the grayscale image
        int grayOffset = y * width + x;
        // one can think of the RGB image having
        // CHANNEL times columns than the gray scale image
        int rgbOffset = grayOffset * channels;
        float r = rgbImage[rgbOffset];     // red value for pixel
        float g = rgbImage[rgbOffset + 1]; // green value for pixel
        float b = rgbImage[rgbOffset + 2]; // blue value for pixel
        // perform the rescaling and store it
        // We multiply by floating point constants
        grayImage[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

int main(int argc, char *argv[])
{

    // check if number of input args is correct: input and output image filename
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input.ppm> <output.ppm>" << std::endl;
        return 0;
    }

    fs::path inputPath = argv[1];
    if (!fs::exists(inputPath))
    {
        std::cerr << "Input file does not exist!" << std::endl;
        return 0;
    }

    Image inputImage = readImageFromPPM(inputPath);
    if (inputImage.isGray())
    {
        std::cerr << "Input image is already grayscale!" << std::endl;
        return 0;
    }

    fs::path outputPath = argv[2];
    if (!fs::exists(outputPath.parent_path()))
    {
        fs::create_directories(outputPath.parent_path());
        std::cout << "Created output directory: " << outputPath.parent_path() << std::endl;
    }

    // NOTE: We create output image explicitly as grayscale
    Image outputImage(inputImage.getWidth(), inputImage.getHeight(), true);

    // allocate input and output images in the device
    float *d_inputImage;
    float *d_outputImage;
    cudaMalloc((void **)&d_inputImage, inputImage.getRows() * inputImage.getCols() * sizeof(float));
    cudaMalloc((void **)&d_outputImage, outputImage.getRows() * outputImage.getCols() * sizeof(float));

    // copy image to the device
    cudaMemcpy(d_inputImage, inputImage.getDataConstPtr(), inputImage.getRows() * inputImage.getCols() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil((float)outputImage.getCols() / TILE_WIDTH), ceil((float)outputImage.getRows() / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    rgb2gray<<<dimGrid, dimBlock>>>(d_outputImage, d_inputImage, 3, outputImage.getCols(), outputImage.getRows());

    cudaMemcpy(outputImage.getDataPtr(), d_outputImage, outputImage.getRows() * outputImage.getCols() * sizeof(float), cudaMemcpyDeviceToHost);
    outputImage.writeToPPM(outputPath);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}
