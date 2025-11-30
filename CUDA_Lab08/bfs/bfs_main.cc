#include "bfs.h"

#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_dir>" << std::endl;
        return EXIT_FAILURE;
    }

    // Check if input file exists
    if (!fs::exists(argv[1]))
    {
        std::cerr << "Input file does not exist: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    // Check if output directory exists - create if it does not
    if (!fs::exists(argv[2]))
    {
        if (!fs::create_directories(argv[2]))
        {
            std::cerr << "Failed to create output directory: " << argv[2] << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Read the graph from the input file
    unsigned int startNode;
    bfs::GraphCSR graph;
    bfs::readTestcaseFromFile(argv[1], graph, startNode);
    if (graph.edges.empty() || graph.dest.empty())
    {
        std::cerr << "Failed to read the graph from file: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    // Run BFS on the host (CPU) with time measurement
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> hostLabels = bfs::bfsOnHost(graph, startNode);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Host BFS time: " << elapsed.count() << " seconds" << std::endl;

    std::ofstream outFile(fs::path(argv[2]) / "bfs_host_labels.txt");
    for (const auto &label : hostLabels)
    {
        outFile << label << std::endl;
    }

    // Warmup run for device BFS
    bfs::bfsOnDevice(graph, startNode, bfs::BFSQueueType::Global);

    // Run BFS on the device (GPU) with Global Queue
    auto startGlobal = std::chrono::high_resolution_clock::now();
    std::vector<int> deviceLabelsGlobal = bfs::bfsOnDevice(graph, startNode, bfs::BFSQueueType::Global);
    auto endGlobal = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedGlobal = endGlobal - startGlobal;
    std::cout << "Device BFS (Global Queue) time: " << elapsedGlobal.count() << " seconds" << std::endl;

    std::ofstream outFileGlobal(fs::path(argv[2]) / "bfs_device_global_labels.txt");
    for (const auto &label : deviceLabelsGlobal)
    {
        outFileGlobal << label << std::endl;
    }

    // Compare w.r.t. host results
    for (size_t i = 0; i < hostLabels.size(); ++i)
    {
        if (hostLabels[i] != deviceLabelsGlobal[i])
        {
            std::cerr << "\033[31mMismatch at node " << i << ": host = " << hostLabels[i]
                      << ", device (global) = " << deviceLabelsGlobal[i] << "\033[0m" << std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "\033[32mDevice BFS (Global Queue) results match host results.\033[0m" << std::endl;

    // Run BFS on the device (GPU) with Block Queue
    auto startBlock = std::chrono::high_resolution_clock::now();
    std::vector<int> deviceLabelsBlock = bfs::bfsOnDevice(graph, startNode, bfs::BFSQueueType::Block);
    auto endBlock = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedBlock = endBlock - startBlock;
    std::cout << "Device BFS (Block Queue) time: " << elapsedBlock.count() << " seconds" << std::endl;

    std::ofstream outFileBlock(fs::path(argv[2]) / "bfs_device_block_labels.txt");
    for (const auto &label : deviceLabelsBlock)
    {
        outFileBlock << label << std::endl;
    }

    // Compare w.r.t. host results
    for (size_t i = 0; i < hostLabels.size(); ++i)
    {
        if (hostLabels[i] != deviceLabelsBlock[i])
        {
            std::cerr << "\033[31mMismatch at node " << i << ": host = " << hostLabels[i]
                      << ", device (block) = " << deviceLabelsBlock[i] << "\033[0m" << std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "\033[32mDevice BFS (Block Queue) results match host results.\033[0m" << std::endl;
    return EXIT_SUCCESS;
}
