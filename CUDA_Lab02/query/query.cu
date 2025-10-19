#include <stdio.h>

int main(int argc, char **argv)
{
  int deviceCount = 0;

  cudaGetDeviceCount(&deviceCount);

  for (int dev = 0; dev < deviceCount; dev++)
  {
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, dev);

    if (dev == 0)
    {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999)
      {
        printf("No CUDA GPU has been detected\n");
        return -1;
      }
      else if (deviceCount == 1)
      {
        printf("There is 1 device supporting CUDA\n");
      }
      else
      {
        printf("There are %d devices supporting CUDA\n", deviceCount);
      }
    }

    printf("\nDevice %d name: %s\n", dev, deviceProp.name);
    printf(" Computational Capabilities: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf(" Maximum global memory size: %lu\n", deviceProp.totalGlobalMem);
    printf(" Maximum constant memory size: %lu\n", deviceProp.totalConstMem);
    printf(" Maximum shared memory size per block: %lu\n", deviceProp.sharedMemPerBlock);
    printf(" Maximum block dimensions: %d x %d x %d\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf(" Maximum grid dimensions: %d x %d x %d\n", deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf(" Warp size: %d\n\n", deviceProp.warpSize);
  }

  return 0;
}
