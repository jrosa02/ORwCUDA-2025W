// Modernized to C++17
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

#define RND_SEED 13 // for tests reproducibility
#define MAX_MASK_VALUE 10
#define MAX_SIGNAL_VALUE 200

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Wrong number of arguments: exactly 3 arguments needed (length of mask, length of signal and output file name)\n";
        return 1;
    }

    // read sizes
    int maskSize = 0;
    int size = 0;
    try {
        maskSize = std::stoi(argv[1]);
        size = std::stoi(argv[2]);
    } catch (const std::exception &e) {
        std::cerr << "Invalid integer argument(s): " << e.what() << '\n';
        return 1;
    }

    // test reproducibility for given size
    const unsigned seed = static_cast<unsigned>(RND_SEED) ^ static_cast<unsigned>(size);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distMask(0.0f, static_cast<float>(MAX_MASK_VALUE));
    std::uniform_real_distribution<float> distSignal(0.0f, static_cast<float>(MAX_SIGNAL_VALUE));

    // open output file
    if (!fs::path(argv[3]).parent_path().empty()) {
        fs::create_directories(fs::path(argv[3]).parent_path());
    }
    std::ofstream ofs(argv[3]);
    if (!ofs)
    {
        std::cerr << argv[3] << ": cannot open output file." << '\n';
        return 2;
    }

    // generate & write data
    ofs << maskSize << ' ' << size << '\n';
    ofs << std::fixed << std::setprecision(2);

    for (int i = 0; i < maskSize; ++i)
    {
        float M = distMask(rng);
        if (i) ofs << ' ';
        ofs << M;
    }
    ofs << '\n';

    for (int i = 0; i < size; ++i)
    {
        float N = distSignal(rng);
        if (i) ofs << ' ';
        ofs << N;
    }

    ofs << '\n';

    return 0;
}