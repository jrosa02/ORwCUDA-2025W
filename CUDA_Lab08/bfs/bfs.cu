#include "bfs.h"

#define CUDA_CHECK(x) do { if ((x) != cudaSuccess) \
    throw std::runtime_error(cudaGetErrorString(x)); } while(0)

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

    __global__ void kernelGlobalQueue(int *edges, int *dest, int *label,
                                  int *pFrontier, int *cFrontier,
                                  int *pFrontierTail, int *cFrontierTail)
    {
        // odczyt rozmiaru poprzedniej frontierki (przechowywany jako jedno int w pamięci device)
        int pSize = 0;
        pSize = *pFrontierTail; // odczyt z pamięci device

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= pSize) return;

        int v = pFrontier[tid];
        int start = edges[v];
        int end = edges[v + 1];
        int myLabel = label[v]; // aktualna odległość wierzchołka v

        for (int ei = start; ei < end; ++ei)
        {
            int nbr = dest[ei];

            // próbujemy atomowo ustawić label[nbr] z -1 na myLabel+1
            int old = atomicCAS(&label[nbr], -1, myLabel + 1);
            if (old == -1)
            {
                // udało się oznaczyć wierzchołek jako odwiedzony — dopisujemy go do cFrontier
                int pos = atomicAdd(cFrontierTail, 1);
                cFrontier[pos] = nbr;
            }
        }
    }


    __global__ void kernelBlockQueue(int *edges, int *dest, int *label, int *pFrontier, int *cFrontier, int *pFrontierTail, int *cFrontierTail)
    {
    }

    std::vector<int> bfsOnDevice(const GraphCSR &graph, unsigned int source, BFSQueueType queueType)
    {
        // nie zmieniamy sygnatury; implementacja korzysta z kolejki globalnej
        (void)queueType; // jeżeli enum jest wymagany, ale w tej funkcji używamy global queue

        const int nodes = static_cast<int>(graph.edges.size() - 1);
        const int edges_size = static_cast<int>(graph.edges.size());
        const int dest_size = static_cast<int>(graph.dest.size());

        // device pointers
        int *d_edges = nullptr;
        int *d_dest = nullptr;
        int *d_label = nullptr;
        int *d_pFrontier = nullptr;
        int *d_cFrontier = nullptr;
        int *d_pFrontierTail = nullptr; // jeden int w device: liczba elementów w pFrontier
        int *d_cFrontierTail = nullptr; // jeden int w device: liczba elementów w cFrontier

        // alokacje
        CUDA_CHECK(cudaMalloc(&d_edges, edges_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_dest, dest_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_label, nodes * sizeof(int)));

        // frontiers: maksymalny rozmiar to nodes
        CUDA_CHECK(cudaMalloc(&d_pFrontier, nodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cFrontier, nodes * sizeof(int)));

        // tail counters (po jednym int na każdy)
        CUDA_CHECK(cudaMalloc(&d_pFrontierTail, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cFrontierTail, sizeof(int)));

        // kopiowanie grafu na device
        CUDA_CHECK(cudaMemcpy(d_edges, graph.edges.data(), edges_size * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dest, graph.dest.data(), dest_size * sizeof(int), cudaMemcpyHostToDevice));

        // ustaw label[] = -1 wszystkich węzłów (cudaMemset ustawia bajty; 0xFF -> -1 dla int)
        CUDA_CHECK(cudaMemset(d_label, 0xFF, nodes * sizeof(int)));

        // przygotowanie początkowej frontierki
        // pFrontier = [source], pFrontierTail = 1, cFrontierTail = 0
        CUDA_CHECK(cudaMemcpy(d_pFrontier, &source, sizeof(int), cudaMemcpyHostToDevice));
        int one = 1;
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_pFrontierTail, &one, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cFrontierTail, &zero, sizeof(int), cudaMemcpyHostToDevice));

        // ustaw label[source] = 0
        CUDA_CHECK(cudaMemcpy(d_label + source, &zero, sizeof(int), cudaMemcpyHostToDevice));

        // główna pętla: host sprawdza, czy poprzednia frontierka nie jest pusta
        int h_pSize = 0;
        while (true)
        {
            // odczytaj rozmiar pFrontier z device
            CUDA_CHECK(cudaMemcpy(&h_pSize, d_pFrontierTail, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_pSize == 0) break;

            // wywołanie kernela: 1 wątek -> 1 element poprzedniej frontierki
            int numBlocks = (h_pSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelGlobalQueue<<<numBlocks, BLOCK_SIZE>>>(d_edges, d_dest, d_label,
                                                        d_pFrontier, d_cFrontier,
                                                        d_pFrontierTail, d_cFrontierTail);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // po kernelu: swap frontiers (tylko wskaźniki) i wyzeruj nową cFrontierTail
            std::swap(d_pFrontier, d_cFrontier);
            std::swap(d_pFrontierTail, d_cFrontierTail);

            // ustaw cFrontierTail (nowy) na 0
            CUDA_CHECK(cudaMemset(d_cFrontierTail, 0, sizeof(int)));
        }

        // kopiujemy label z powrotem na host
        std::vector<int> h_label(nodes);
        CUDA_CHECK(cudaMemcpy(h_label.data(), d_label, nodes * sizeof(int), cudaMemcpyDeviceToHost));

        // czyszczenie
        cudaFree(d_edges);
        cudaFree(d_dest);
        cudaFree(d_label);
        cudaFree(d_pFrontier);
        cudaFree(d_cFrontier);
        cudaFree(d_pFrontierTail);
        cudaFree(d_cFrontierTail);

        return h_label;
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
