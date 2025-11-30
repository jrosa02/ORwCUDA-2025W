#include "bfs.h"

namespace bfs
{
    void readTestcaseFromFile(const fs::path &filePath, GraphCSR &graph, unsigned int &startNode)
    {
        std::ifstream file(filePath);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file: " + filePath.string());
        }

        // Read source node idx
        file >> startNode;

        // Read edges array
        int numEdges;
        file >> numEdges;
        graph.edges.resize(numEdges);
        for (int i = 0; i < numEdges; ++i)
        {
            file >> graph.edges[i];
        }

        // Read dest array
        int numDest;
        file >> numDest;
        graph.dest.resize(numDest);
        for (int i = 0; i < numDest; ++i)
        {
            file >> graph.dest[i];
        }
    }

    __global__ void kernelGlobalQueue(int *edges, int *dest, int *label, int *pFrontier, int *cFrontier, int *pFrontierTail, int *cFrontierTail)
    {
    }

    __global__ void kernelBlockQueue(int *edges, int *dest, int *label, int *pFrontier, int *cFrontier, int *pFrontierTail, int *cFrontierTail)
    {
    }

    std::vector<int> bfsOnDevice(const GraphCSR &graph, unsigned int source, BFSQueueType queueType)
    {
    }

    std::vector<int> bfsOnHost(const GraphCSR &graph, unsigned int source)
    {
        int nodes = static_cast<int>(graph.edges.size() - 1);
        std::vector<int> label(nodes, -1);
        label[source] = 0;

        std::vector<int> pFrontier;
        pFrontier.push_back(source);

        while (!pFrontier.empty())
        {
            std::vector<int> cFrontier;
            for (const auto &cVertex : pFrontier)
            {
                for (int i = graph.edges[cVertex]; i < graph.edges[cVertex + 1]; ++i)
                {
                    int neighbor = graph.dest[i];
                    if (label[neighbor] == -1)
                    {
                        label[neighbor] = label[cVertex] + 1;
                        cFrontier.push_back(neighbor);
                    }
                }
            }
            pFrontier.swap(cFrontier);
        }

        return label;
    }
} // namespace bfs
