#include <iostream>
#include <filesystem>
#include <chrono>
#include "add_mm.h"

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
        Matrix A = readFromFile(fileA);
        Matrix B = readFromFile(fileB);
        if (A.getRows() != B.getRows() || A.getCols() != B.getCols())
        {
            throw std::runtime_error("Matrices must have the same dimensions for addition.");
        }
        std::cout << "Adding matrices..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        Matrix C0 = addMatricesOnHost(A, B);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "\033[35mHost addition completed in " << duration.count() << " ms.\033[0m" << std::endl;
        fs::path outFile0 = outputDir / "reference_result.txt";
        C0.writeToFile(outFile0);
        std::cout << "Reference result written to " << outFile0 << std::endl;

        // Additional GPU computation to "warm up" the GPU
        addMatricesOnDevice(A, B, AddMethod::ByElements);

        start = std::chrono::high_resolution_clock::now();
        Matrix C1 = addMatricesOnDevice(A, B, AddMethod::ByElements);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        std::cout << "\033[35mDevice addition (by elements) completed in " << duration.count() << " ms.\033[0m" << std::endl;
        fs::path outFile1 = outputDir / "result_by_elements.txt";
        C1.writeToFile(outFile1);
        std::cout << "Result by elements written to " << outFile1 << std::endl;

        start = std::chrono::high_resolution_clock::now();
        Matrix C2 = addMatricesOnDevice(A, B, AddMethod::ByRows);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        std::cout << "\033[35mDevice addition (by rows) completed in " << duration.count() << " ms.\033[0m" << std::endl;
        fs::path outFile2 = outputDir / "result_by_rows.txt";
        C2.writeToFile(outFile2);
        std::cout << "Result by rows written to " << outFile2 << std::endl;

        start = std::chrono::high_resolution_clock::now();
        Matrix C3 = addMatricesOnDevice(A, B, AddMethod::ByColumns);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        std::cout << "\033[35mDevice addition (by columns) completed in " << duration.count() << " ms.\033[0m" << std::endl;
        fs::path outFile3 = outputDir / "result_by_cols.txt";
        C3.writeToFile(outFile3);
        std::cout << "Result by columns written to " << outFile3 << std::endl;
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}