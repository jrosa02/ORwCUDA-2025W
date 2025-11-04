#include "conv_2d.h"
#include "image.h"

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <mask_size> <input_file> <output_dir>" << std::endl;
        return 1;
    }

    int maskSize = std::stoi(argv[1]);
    fs::path inputFile = argv[2];
    fs::path outputDir = argv[3];

    if (!fs::exists(inputFile))
    {
        std::cerr << "Input file does not exist: " << inputFile << std::endl;
        return 1;
    }

    if (!fs::exists(outputDir))
    {
        fs::create_directories(outputDir);
    }

    // Read input image
    Image img = readImageFromPPM(inputFile);

    // Define a simple edge detection mask
    std::vector<float> mask;
    for (int i = 0; i < maskSize * maskSize; ++i)
    {
        mask.push_back(1.0f / (float)(maskSize * maskSize));
    }

    std::cout << "Performing 2D convolution..." << std::endl;
    std::cout << "Image resolution: " << img.getWidth() << "x" << img.getHeight() << std::endl;
    std::cout << "Mask size: " << maskSize << "x" << maskSize << std::endl;

    // Perform convolution on host
    auto start = std::chrono::high_resolution_clock::now();
    Image resultHost = convolutionOnHost(img, mask);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "\033[35mHost computation time: " << duration.count() << " seconds.\033[0m" << std::endl;

    fs::path outputFileHost = outputDir / "convolution_host.ppm";
    resultHost.writeToPPM(outputFileHost);
    std::cout << "Result (host) written to " << outputFileHost << std::endl;

    // GPU warm-up
    convolutionOnDevice(img, mask, ConvMethod::Basic);

    // Perform convolution on device
    start = std::chrono::high_resolution_clock::now();
    Image resultDevice = convolutionOnDevice(img, mask, ConvMethod::Basic);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "\033[35mDevice (basic) computation time: " << duration.count() << " seconds.\033[0m" << std::endl;
    fs::path outputFileDevice = outputDir / "convolution_device_basic.ppm";
    resultDevice.writeToPPM(outputFileDevice);
    std::cout << "Result (device - basic) written to " << outputFileDevice << std::endl;

    start = std::chrono::high_resolution_clock::now();
    resultDevice = convolutionOnDevice(img, mask, ConvMethod::Tiled);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "\033[35mDevice (tiled) computation time: " << duration.count() << " seconds.\033[0m" << std::endl;
    outputFileDevice = outputDir / "convolution_device_tiled.ppm";
    resultDevice.writeToPPM(outputFileDevice);
    std::cout << "Result (device - tiled) written to " << outputFileDevice << std::endl;
    return 0;
}
