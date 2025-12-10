#include "histogram.h"
#include "histogramUtils.h"

#include <iostream>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_dir> <nBins>" << std::endl;
        return 1;
    }

    fs::path inputFile = argv[1];
    fs::path outputDir = argv[2];
    int nBins = std::stoi(argv[3]);

    if (nBins <= 0 || nBins > N_LETTERS)
    {
        std::cerr << "nBins must be in the range [1, " << N_LETTERS << "]" << std::endl;
        return 1;
    }

    if (!fs::exists(inputFile))
    {
        std::cerr << "Input file does not exist: " << inputFile << std::endl;
        return 1;
    }

    if (!fs::exists(outputDir))
    {
        fs::create_directories(outputDir);
    }

    // Read input file
    auto fileData = histogram_utils::readFile(inputFile);

    // Compute histogram on host
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<unsigned int> histogram = computeHistogramOnHost(fileData, nBins);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "\033[35mHost computation time: " << duration.count() << " seconds\033[0m" << std::endl;

    fs::path outputFile = outputDir / "host.txt";
    histogram_utils::writeHistogram(outputFile, histogram, nBins);

    // GPU warm-up
    computeHistogramOnDevice(fileData, nBins, HistMethod::Block);

    // Compute histogram on device
    start = std::chrono::high_resolution_clock::now();
    histogram = computeHistogramOnDevice(fileData, nBins, HistMethod::Block);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "\033[35mDevice computation time (Block): " << duration.count() << " seconds\033[0m" << std::endl;

    outputFile = outputDir / "block.txt";
    histogram_utils::writeHistogram(outputFile, histogram, nBins);

    // Compute histogram on device - Interleaved
    start = std::chrono::high_resolution_clock::now();
    histogram = computeHistogramOnDevice(fileData, nBins, HistMethod::Interleaved);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "\033[35mDevice computation time (Interleaved): " << duration.count() << " seconds\033[0m" << std::endl;

    outputFile = outputDir / "interleaved.txt";
    histogram_utils::writeHistogram(outputFile, histogram, nBins);

    // Compute histogram on device - Privatisation
    start = std::chrono::high_resolution_clock::now();
    histogram = computeHistogramOnDevice(fileData, nBins, HistMethod::Privatised);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "\033[35mDevice computation time (Privatisation): " << duration.count() << " seconds\033[0m" << std::endl;

    outputFile = outputDir / "privatisation.txt";
    histogram_utils::writeHistogram(outputFile, histogram, nBins);

    return 0;
}
