#include <iostream>
#include <fstream>
#include <filesystem>
#include <ctime>
#include <matrix.h>

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <rows> <cols> <output_dir>" << std::endl;
        return 1;
    }

    unsigned int rows = std::stoi(argv[1]);
    unsigned int cols = std::stoi(argv[2]);
    fs::path outputDir = argv[3];

    if (!fs::exists(outputDir))
    {
        fs::create_directories(outputDir);
    }

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    Matrix matA = generateRandomMatrix(rows, cols);
    fs::path outputPath = outputDir / "mat_A.txt";
    matA.writeToFile(outputPath);

    Matrix matB = generateRandomMatrix(rows, cols);
    outputPath = outputDir / "mat_B.txt";
    matB.writeToFile(outputPath);

    std::cout << "Testcase for matrices of size (" << rows << ", " << cols << ") written to " << outputDir << std::endl;
    return 0;
}