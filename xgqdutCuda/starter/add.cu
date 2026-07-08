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

int main() {
    float *hostA, *hostB, *hostC, *serialC;  // serialC is calculated with cpu adding
    int n = 10 * 1024;

    hostA = (float *)malloc(n * sizeof(float));
    hostB = (float *)malloc(n * sizeof(float));
    hostC = (float *)malloc(n * sizeof(float));
    serialC = (float *)malloc(n * sizeof(float));

    initCpu(hostA, hostB, n);
    addCpu(hostA, hostB, serialC, n);


    return 0;
}
