#include <iostream>
#include <filesystem>
#include <fstream>

#include "convFiles.h"

namespace fs = std::filesystem;

std::vector<float> readConvolutionOutput(const fs::path &filePath)
{
    std::vector<float> output;
    std::ifstream inputFile(filePath);
    if (!inputFile)
    {
        std::cerr << "Error reading file: " << filePath << std::endl;
        return output;
    }
    float value;
    while (inputFile >> value)
    {
        output.push_back(value);
    }
    return output;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_dir>" << std::endl;
        return 1;
    }

    fs::path inputFilePath = argv[1];
    fs::path outputDir = argv[2];

    if (!fs::exists(inputFilePath))
    {
        std::cerr << "Input file does not exist: " << inputFilePath << std::endl;
        return 1;
    }

    if (!fs::exists(outputDir))
    {
        std::cerr << "Output directory does not exist: " << outputDir << std::endl;
        return 1;
    }

    // Read input data lengths
    int maskSize = 0, signalSize = 0;
    if (conv::getSizes1D(inputFilePath.string(), maskSize, signalSize) != conv::FileStatus::Success)
    {
        std::cerr << "Error reading input file: " << inputFilePath << std::endl;
        return 1;
    }

    // Read baseline output (host)
    fs::path baselineFile = outputDir / "convolution_host.txt";
    std::vector<float> baselineOutput = readConvolutionOutput(baselineFile);

    // Read and verify device outputs
    fs::path deviceFile = outputDir / "convolution_device_basic.txt";
    std::vector<float> deviceOutput = readConvolutionOutput(deviceFile);
    if (deviceOutput.size() != signalSize)
    {
        std::cerr << "\033[31m[ERROR] Output size mismatch: expected " << signalSize
                  << ", got " << deviceOutput.size() << "\033[0m" << std::endl;
        return 1;
    }
    const float epsilon = 1e-1f;
    for (std::size_t i = 0; i < baselineOutput.size(); ++i)
    {
        if (std::abs(baselineOutput[i] - deviceOutput[i]) > epsilon)
        {
            std::cerr << "\033[31m[ERROR] Device basic mismatch at index " << i << ": baseline = " << baselineOutput[i]
                      << ", device = " << deviceOutput[i] << "\033[0m" << std::endl;
            return 1;
        }
    }

    deviceFile = outputDir / "convolution_device_tiled.txt";
    deviceOutput = readConvolutionOutput(deviceFile);
    if (deviceOutput.size() != signalSize)
    {
        std::cerr << "\033[31m[ERROR] Output size mismatch: expected " << signalSize
                  << ", got " << deviceOutput.size() << "\033[0m" << std::endl;
        return 1;
    }
    for (std::size_t i = 0; i < baselineOutput.size(); ++i)
    {
        if (std::abs(baselineOutput[i] - deviceOutput[i]) > epsilon)
        {
            std::cerr << "\033[31m[ERROR] Device tiled mismatch at index " << i << ": baseline = " << baselineOutput[i]
                      << ", device = " << deviceOutput[i] << "\033[0m" << std::endl;
            return 1;
        }
    }

    std::cout << "\e[1m\033[32mTests passed!\033[0m\e[0m" << std::endl;

    return 0;
}