#ifndef BFS_H_
#define BFS_H_

#include <vector>
#include <fstream>
#include <filesystem>

#define BLOCK_SIZE 256
#define BLOCK_QUEUE_SIZE 8192

namespace fs = std::filesystem;

namespace bfs
{
    struct GraphCSR
    {
        // Starting indices of edges for each node
        std::vector<int> edges;

        // Destination nodes for each edge
        std::vector<int> dest;
    };

    enum class BFSQueueType
    {
        Global,
        Block
    };

    /**
     * @brief Reads a graph and starting node from a file.
     *
     * @param filePath Path to the input file
     * @param graph Output graph in CSR format
     * @param startNode Output starting node index
     */
    void readTestcaseFromFile(const fs::path &filePath, GraphCSR &graph, unsigned int &startNode);

    /**
     * @brief Performs BFS on the host (CPU).
     *
     * @param graph Input graph in CSR format
     * @param startNode Index of the starting node
     * @return std::vector<int> Vector containing distances from the start node (labels)
     */
    std::vector<int> bfsOnHost(const GraphCSR &graph, unsigned int startNode);

    /**
     * @brief Performs BFS on the device (GPU).
     *
     * @param graph Input graph in CSR format
     * @param startNode Index of the starting node
     * @param queueType Type of queue to use (default: Global)
     * @return std::vector<int> Vector containing distances from the start node (labels)
     */
    std::vector<int> bfsOnDevice(const GraphCSR &graph, unsigned int startNode, BFSQueueType queueType = BFSQueueType::Global);
} // namespace bfs

#endif // BFS_H_
