#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>

void initCpu(float *hostA, float *hostB, int n) {
    for (int i = 0; i < n; ++i) {
        hostA[i] = 1;
        hostB[i] = 1;
    }
}

void addCpu(float *hostA, float *hostB, float *hostC, int n)
{
    for (int i = 0; i < n; i++)
    {
        hostC[i] = hostA[i] + hostB[i];
    }
}

__global__ void addKernel(float *deviceA, float *deviceB, float *deviceC, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        deviceC[idx] = deviceA[idx] + deviceB[idx]; 
    }
}

int main() {
    float *hostA, *hostB, *hostC, *serialC;  // serialC is calculated with cpu adding
    int n = 10 * 1024;

    hostA = (float *)malloc(n * sizeof(float));
    hostB = (float *)malloc(n * sizeof(float));
    hostC = (float *)malloc(n * sizeof(float));
    serialC = (float *)malloc(n * sizeof(float));

    initCpu(hostA, hostB, n);
    addCpu(hostA, hostB, serialC, n);

    float *deviceA, *deviceB, *deviceC;
    cudaMalloc((void **)&deviceA, n * sizeof(float));
    cudaMalloc((void **)&deviceB, n * sizeof(float));
    cudaMalloc((void **)&deviceC, n * sizeof(float));

    //printf("address of hostA: %p\n", &hostA);
    //printf("address of deviceA: %p\n", &deviceA);

    //printf("value of hostA: %p\n", hostA);
    //printf("value of deviceA: %p\n", deviceA);

    //printf("value of hostA[0]: %f\n", hostA[0]);
    //printf("value of deviceA[0]: %f\n", deviceA[0]);

    cudaMemcpy(deviceA, hostA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, n * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int BLOCK_DIM = 256;
    unsigned int GRID_DIM = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 grid_dim = {GRID_DIM, 1, 1};
    dim3 block_dim = {BLOCK_DIM, 1, 1};

    addKernel<<<grid_dim, block_dim>>>(deviceA, deviceB, deviceC, n);
    cudaDeviceSynchronize();
    cudaMemcpy(hostC, deviceC, n * sizeof(float), cudaMemcpyDeviceToHost);

    // compare the results
    for (int i = 0; i < n; i++) {
        if (hostC[i] != serialC[i]) {
            printf("Error: hostC[%d] = %f, serialC[%d] = %f\n", i, hostC[i], i, serialC[i]);
            break;
        }
    }

    // Cleanup
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}
