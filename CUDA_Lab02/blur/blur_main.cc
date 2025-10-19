#include <iostream>
#include <filesystem>
#include <chrono>
#include "blur.h"

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_dir> <kernel_size>" << std::endl;
        return 1;
    }

    fs::path inputFilePath = argv[1];
    fs::path outputDirectory = argv[2];
    int kernelSize = std::stoi(argv[3]);

    if (kernelSize % 2 == 0 || kernelSize < 1)
    {
        std::cerr << "Kernel size must be a positive odd integer." << std::endl;
        return 1;
    }

    if (!fs::exists(inputFilePath))
    {
        std::cerr << "Input file does not exist: " << inputFilePath << std::endl;
        return 1;
    }

    if (!fs::exists(outputDirectory))
    {
        fs::create_directories(outputDirectory);
        std::cout << "Created output directory: " << outputDirectory << std::endl;
    }

    std::cout << "Reading input image..." << std::endl;
    Image inputImage = readImageFromPPM(inputFilePath);

    std::cout << "Applying blur with kernel size " << kernelSize << "..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    Image outputImage = imageBlurOnHost(inputImage, kernelSize);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "\033[35mHost blur completed in " << duration.count() << " ms.\033[0m" << std::endl;
    fs::path outputFilePath = outputDirectory / "reference_image.ppm";
    outputImage.writeToPPM(outputFilePath);
    std::cout << "Reference image written to " << outputFilePath << std::endl;

    // Additional GPU computation to "warm up" the GPU
    imageBlurOnDevice(inputImage, kernelSize);

    start = std::chrono::high_resolution_clock::now();
    outputImage = imageBlurOnDevice(inputImage, kernelSize);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "\033[35mDevice blur completed in " << duration.count() << " ms.\033[0m" << std::endl;
    outputFilePath = outputDirectory / "result_image.ppm";
    outputImage.writeToPPM(outputFilePath);
    std::cout << "Result image written to " << outputFilePath << std::endl;

    std::cout << "Done." << std::endl;
    return 0;
}
