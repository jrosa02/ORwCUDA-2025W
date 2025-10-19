#include <iostream>
#include <filesystem>
#include <ctime>
#include <random>
#include <matrix.h>

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <rows (output_dim)> <cols (input_dim)> <output_dir>" << std::endl;
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

    Matrix matX = generateRandomMatrix(cols, 1);
    outputPath = outputDir / "vec_x.txt";
    matX.writeToFile(outputPath);

    std::cout << "Testcase for matrix of size (" << rows << ", " << cols << ") and vector of size (" << cols << ", 1) written to " << outputDir << std::endl;
    return 0;
}