#include "check_error.h"

#define cudaCheckError()                                                                     \
    {                                                                                        \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess)                                                                \
        {                                                                                    \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    }

__global__ void kernel(int *a, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = i;
}

void check_error()
{

    int N = 4097;
    int threads = 128;
    int blocks = (N + threads - 1) / threads;
    printf("%d, %d", N, threads*blocks);
    int *a = NULL;

    cudaMallocManaged(&a, blocks*threads*sizeof(int));
    kernel<<<blocks, threads>>>(a, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < 10; i++)
        std::cout << a[i] << std::endl;

    cudaFree(a);

    cudaCheckError();
}
