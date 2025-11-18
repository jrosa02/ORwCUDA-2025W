#include "scan.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <n_bits>" << std::endl;
        return 1;
    }

    // Create input data vector
    int n_bits = 0;
    try
    {
        n_bits = std::stoi(argv[1]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Invalid integer argument(s): " << e.what() << '\n';
        return 1;
    }
    std::vector<int> data(1 << n_bits, 1);

    std::cout << "Performing scan..." << std::endl;
    std::cout << "Data size: " << data.size() << std::endl;
    std::cout << std::scientific << std::setprecision(6);

    // Perform scan on host
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> resultHost = scanOnHost(data);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "\033[35mHost computation time: " << duration.count() << " seconds.\033[0m" << std::endl;

    // GPU warm-up
    scanOnDevice(data, ScanMethod::KoggeStone);

    // Perform scan on device
    start = std::chrono::high_resolution_clock::now();
    std::vector<int> resultDevice = scanOnDevice(data, ScanMethod::KoggeStone);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "\033[35mDevice (KoggeStone) computation time: " << duration.count() << " seconds.\033[0m" << std::endl;

    // Iterate to find differences
    bool match = true;
    for (size_t i = 0; i < resultHost.size(); ++i)
    {
        if (resultHost[i] != resultDevice[i])
        {
            std::cerr << "\033[31mMismatch at index " << i << ": host = " << resultHost[i]
                      << ", device = " << resultDevice[i] << "\033[0m" << std::endl;
            match = false;
            break;
        }
    }
    if (match)
    {
        std::cout << "\033[32mResults match!\033[0m" << std::endl;
    }

    return 0;
}