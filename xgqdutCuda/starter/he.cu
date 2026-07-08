#include <cuda.h>
#include <stdio.h>

__global__ void printHello() {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    printf("hello world from GPU by thread: %d\n", index);
}

int main() {
    dim3 grid_dim = {1, 1, 1};
    dim3 block_dim = {4, 1, 1};

    printHello<<<grid_dim, block_dim>>>();

    cudaDeviceSynchronize();

    return 0;
}
