#include <iostream>
#include <filesystem>
#include <chrono>
#include "mult_mm.h"

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <matrix_file_A> <matrix_file_B> <output_dir>" << std::endl;
        return 1;
    }

    fs::path fileA = argv[1];
    fs::path fileB = argv[2];
    fs::path outputDir = argv[3];

    if (!fs::exists(outputDir))
    {
        fs::create_directories(outputDir);
        std::cout << "Created output directory: " << outputDir << std::endl;
    }

    std::cout << "Reading matrices from files..." << std::endl;
    try
    {
        // Load matrices from files
        Matrix A = readFromFile(fileA);
        Matrix B = readFromFile(fileB);

        // Perform matrix multiplication on host
        std::cout << "Performing matrix multiplication on host..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        Matrix C_host = multMatrixMatrixOnHost(A, B);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "\033[35mHost computation time: " << duration.count() << " seconds.\033[0m" << std::endl;
        fs::path outputPath = outputDir / "C_host.txt";
        C_host.writeToFile(outputPath);
        std::cout << "Reference (host) matrix written to " << outputPath << std::endl;

        // Additional GPU computation to "warm up" the GPU
        multMatrixMatrixOnDevice(A, B, MultMethod::Standard);

        // Perform matrix multiplication on device without tiling
        std::cout << "Performing matrix multiplication on device without tiling..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        Matrix C_device = multMatrixMatrixOnDevice(A, B, MultMethod::Standard);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        std::cout << "\033[35mDevice computation time: " << duration.count() << " seconds.\033[0m" << std::endl;
        outputPath = outputDir / "C_device.txt";
        C_device.writeToFile(outputPath);
        std::cout << "Result (device, no tiling) matrix written to " << outputPath << std::endl;

        // Perform matrix multiplication on device with tiling
        std::cout << "Performing matrix multiplication on device with tiling..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        C_device = multMatrixMatrixOnDevice(A, B, MultMethod::Tiled);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        std::cout << "\033[35mDevice (tiled) computation time: " << duration.count() << " seconds.\033[0m" << std::endl;
        outputPath = outputDir / "C_device_tiled.txt";
        C_device.writeToFile(outputPath);
        std::cout << "Result (device, tiled) matrix written to " << outputPath << std::endl;

        // Perform matrix multiplication on device with granular tiling
        std::cout << "Performing matrix multiplication on device with granular tiling..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        C_device = multMatrixMatrixOnDevice(A, B, MultMethod::Granular);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        std::cout << "\033[35mDevice (granular) computation time: " << duration.count() << " seconds.\033[0m" << std::endl;
        outputPath = outputDir / "C_device_granular.txt";
        C_device.writeToFile(outputPath);
        std::cout << "Result (device, granular) matrix written to " << outputPath << std::endl;
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}