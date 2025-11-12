#include "reduction.h"

#include <filesystem>
#include <iostream>
#include <fstream>

namespace fs = std::filesystem;

const double EPS = 1e-5;

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

    std::cout << "Performing reduction..." << std::endl;
    std::cout << "Data size: " << data.size() << std::endl;
    std::cout << std::scientific << std::setprecision(6);

    // Perform reduction on host
    auto start = std::chrono::high_resolution_clock::now();
    int resultHost = reductionOnHost(data);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "\033[35mHost computation time: " << duration.count() << " seconds.\033[0m" << std::endl;

    // GPU warm-up
    reductionOnDevice(data, ReductionMethod::Basic);

    // Perform reduction on device
    start = std::chrono::high_resolution_clock::now();
    int resultDevice = reductionOnDevice(data, ReductionMethod::Basic);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "\033[35mDevice computation time: " << duration.count() << " seconds.\033[0m" << std::endl;
    if (std::abs(resultDevice - resultHost) > EPS)
    {
        std::cerr << "\033[31mError: Device (basic) result does not match host result!\033[0m" << std::endl;
        std::cerr << "Host result: " << resultHost << ", Device result: " << resultDevice << std::endl;
    }
    else
    {
        std::cout << "\033[32mSuccess: Device (basic) result matches host result.\033[0m" << std::endl;
    }

    start = std::chrono::high_resolution_clock::now();
    resultDevice = reductionOnDevice(data, ReductionMethod::Optimized);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "\033[35mDevice (optimized) computation time: " << duration.count() << " seconds.\033[0m" << std::endl;
    if (std::abs(resultDevice - resultHost) > EPS)
    {
        std::cerr << "\033[31mError: Device (optimized) result does not match host result!\033[0m" << std::endl;
        std::cerr << "Host result: " << resultHost << ", Device result: " << resultDevice << std::endl;
    }
    else
    {
        std::cout << "\033[32mSuccess: Device (optimized) result matches host result.\033[0m" << std::endl;
    }

    // start = std::chrono::high_resolution_clock::now();
    // resultDevice = reductionOnDevice(data, ReductionMethod::CooperativeGroups);
    // end = std::chrono::high_resolution_clock::now();
    // duration = end - start;
    // std::cout << "\033[35mDevice (cooperative groups) computation time: " << duration.count() << " seconds.\033[0m" << std::endl;
    // if (std::abs(resultDevice - resultHost) > EPS)
    // {
    //     std::cerr << "\033[31mError: Device (cooperative groups) result does not match host result!\033[0m" << std::endl;
    //     std::cerr << "Host result: " << resultHost << ", Device result: " << resultDevice << std::endl;
    // }
    // else
    // {
    //     std::cout << "\033[32mSuccess: Device (cooperative groups) result matches host result.\033[0m" << std::endl;
    // }

    return 0;
}
