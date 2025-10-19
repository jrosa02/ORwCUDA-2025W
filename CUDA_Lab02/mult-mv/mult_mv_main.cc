#include <iostream>
#include <filesystem>
#include <chrono>
#include "mult_mv.h"

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <matrix_file_A> <vector_file_x> <output_dir>" << std::endl;
        return 1;
    }

    fs::path fileA = argv[1];
    fs::path fileX = argv[2];
    fs::path outputDir = argv[3];

    if (!fs::exists(outputDir))
    {
        fs::create_directories(outputDir);
        std::cout << "Created output directory: " << outputDir << std::endl;
    }

    std::cout << "Reading matrix and vector from files..." << std::endl;
    try
    {
        Matrix A = readFromFile(fileA);
        Matrix x = readFromFile(fileX);
        if (A.getCols() != x.getRows() || x.getCols() != 1)
        {
            throw std::runtime_error("Matrix columns must match vector rows, and vector must be a column vector.");
        }
        std::cout << "Multiplying matrix and vector..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        Matrix b = multMatrixVectorOnHost(A, x);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "\033[35mHost multiplication completed in " << duration.count() << " ms.\033[0m" << std::endl;
        fs::path outputPath = outputDir / "reference_vec_b.txt";
        b.writeToFile(outputPath);
        std::cout << "Reference vector written to " << outputPath << std::endl;

        // Additional GPU computation to "warm up" the GPU
        multMatrixVectorOnDevice(A, x);

        start = std::chrono::high_resolution_clock::now();
        b = multMatrixVectorOnDevice(A, x);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        std::cout << "\033[35mDevice multiplication completed in " << duration.count() << " ms.\033[0m" << std::endl;
        outputPath = outputDir / "result_vec_b.txt";
        b.writeToFile(outputPath);
        std::cout << "Resulting vector written to " << outputPath << std::endl;
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}