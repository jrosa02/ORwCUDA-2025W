#include "conv_1d.h"
#include "convFiles.h"

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_dir>" << std::endl;
        return 1;
    }

    fs::path inputFile = argv[1];
    fs::path outputDir = argv[2];

    if (!fs::exists(inputFile))
    {
        std::cerr << "Input file does not exist: " << inputFile << std::endl;
        return 1;
    }

    if (!fs::exists(outputDir))
    {
        fs::create_directories(outputDir);
    }

    // Read input data
    std::vector<float> mask, signal;
    if (conv::getValues1D(inputFile.string(), mask, signal) != conv::FileStatus::Success)
    {
        std::cerr << "Error reading input file: " << inputFile << std::endl;
        return 1;
    }

    std::cout << "Performing 1D convolution..." << std::endl;
    std::cout << "Signal size: " << signal.size() << std::endl;
    std::cout << "Mask size: " << mask.size() << std::endl;

    // Perform convolution on host
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> resultHost = convolutionOnHost(signal, mask);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "\033[35mHost computation time: " << duration.count() << " seconds.\033[0m" << std::endl;

    fs::path outputFileHost = outputDir / "convolution_host.txt";
    conv::writeData1D(outputFileHost.string(), resultHost);
    std::cout << "Result (host) written to " << outputFileHost << std::endl;

    // GPU warm-up
    convolutionOnDevice(signal, mask, ConvMethod::Basic);

    // Perform convolution on device
    start = std::chrono::high_resolution_clock::now();
    std::vector<float> resultDevice = convolutionOnDevice(signal, mask, ConvMethod::Basic);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "\033[35mDevice (basic) computation time: " << duration.count() << " seconds.\033[0m" << std::endl;
    fs::path outputFileDevice = outputDir / "convolution_device_basic.txt";
    conv::writeData1D(outputFileDevice.string(), resultDevice);
    std::cout << "Result (device - basic) written to " << outputFileDevice << std::endl;

    start = std::chrono::high_resolution_clock::now();
    resultDevice = convolutionOnDevice(signal, mask, ConvMethod::Tiled);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "\033[35mDevice (tiled) computation time: " << duration.count() << " seconds.\033[0m" << std::endl;
    outputFileDevice = outputDir / "convolution_device_tiled.txt";
    conv::writeData1D(outputFileDevice.string(), resultDevice);
    std::cout << "Result (device - tiled) written to " << outputFileDevice << std::endl;
    return 0;
}
