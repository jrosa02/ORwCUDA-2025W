#include <iostream>
#include <filesystem>
#include <ctime>
#include <random>
#include <matrix.h>

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <A_rows> <A_cols> <B_cols> <output_dir>" << std::endl;
        return 1;
    }

    unsigned int A_rows = std::stoi(argv[1]);
    unsigned int A_cols = std::stoi(argv[2]);
    unsigned int B_cols = std::stoi(argv[3]);
    fs::path outputDir = argv[4];

    if (!fs::exists(outputDir))
    {
        fs::create_directories(outputDir);
    }

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    Matrix matA = generateRandomMatrix(A_rows, A_cols);
    fs::path outputPath = outputDir / "mat_A.txt";
    matA.writeToFile(outputPath);

    Matrix matB = generateRandomMatrix(A_cols, B_cols);
    outputPath = outputDir / "mat_B.txt";
    matB.writeToFile(outputPath);

    std::cout << "Testcase for matrices of size (" << A_rows << ", " << A_cols << ") and (" << A_cols << ", " << B_cols << ") written to " << outputDir << std::endl;
    return 0;
}