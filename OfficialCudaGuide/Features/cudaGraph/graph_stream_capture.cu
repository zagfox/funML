#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256

// ============================================================
// Host callback: fires after the graph completes
// ============================================================
typedef struct {
    void  *data;
    char  *fn_name;
} callBackData_t;

static void myHostNodeCallback(void *userData) {
    callBackData_t *hostFnData = (callBackData_t *)userData;
    double result = *(double *)hostFnData->data;
    printf("[%s] Final result: %f\n", hostFnData->fn_name, result);
}

// ============================================================
// Stage 1: block-level parallel reduction
// ============================================================
__global__ void reduce(const float *input, double *output,
                       size_t inputSize, size_t numOfBlocks) {
    extern __shared__ double smem[];
    size_t tid       = threadIdx.x;
    size_t globalId  = blockIdx.x * blockDim.x + tid;
    size_t gridSize  = numOfBlocks * blockDim.x;

    double sum = 0.0;
    for (size_t i = globalId; i < inputSize; i += gridSize)
        sum += (double)input[i];

    smem[tid] = sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = smem[0];
}

// ============================================================
// Stage 2: final reduction of block partial sums into a scalar
// ============================================================
__global__ void reduceFinal(const double *input, double *output,
                            size_t numOfBlocks) {
    extern __shared__ double smem[];
    size_t tid = threadIdx.x;

    double sum = 0.0;
    for (size_t i = tid; i < numOfBlocks; i += blockDim.x)
        sum += input[i];

    smem[tid] = sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) *output = smem[0];
}

// ============================================================
// Build a graph via stream capture (the function you provided)
// ============================================================
void cudaGraphsUsingStreamCapture(float  *inputVec_h,
                                  float  *inputVec_d,
                                  double *outputVec_d,
                                  double *result_d,
                                  size_t  inputSize,
                                  size_t  numOfBlocks)
{
    cudaStream_t stream1, stream2, stream3, streamForGraph;
    cudaEvent_t  forkStreamEvent, memsetEvent1, memsetEvent2;
    cudaGraph_t  graph;
    double       result_h = 0.0;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&streamForGraph);

    cudaEventCreate(&forkStreamEvent);
    cudaEventCreate(&memsetEvent1);
    cudaEventCreate(&memsetEvent2);

    cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

    cudaEventRecord(forkStreamEvent, stream1);
    cudaStreamWaitEvent(stream2, forkStreamEvent, 0);
    cudaStreamWaitEvent(stream3, forkStreamEvent, 0);

    cudaMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize,
                    cudaMemcpyDefault, stream1);

    cudaMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks, stream2);

    cudaEventRecord(memsetEvent1, stream2);

    cudaMemsetAsync(result_d, 0, sizeof(double), stream3);
    cudaEventRecord(memsetEvent2, stream3);

    cudaStreamWaitEvent(stream1, memsetEvent1, 0);

    reduce<<<numOfBlocks, THREADS_PER_BLOCK,
             THREADS_PER_BLOCK * sizeof(double), stream1>>>(
        inputVec_d, outputVec_d, inputSize, numOfBlocks);

    cudaStreamWaitEvent(stream1, memsetEvent2, 0);

    reduceFinal<<<1, THREADS_PER_BLOCK,
                  THREADS_PER_BLOCK * sizeof(double), stream1>>>(
        outputVec_d, result_d, numOfBlocks);
    cudaMemcpyAsync(&result_h, result_d, sizeof(double),
                    cudaMemcpyDefault, stream1);

    callBackData_t hostFnData = {0};
    hostFnData.data           = &result_h;
    hostFnData.fn_name        = (char *)"cudaGraphsUsingStreamCapture";
    cudaHostFn_t fn           = myHostNodeCallback;
    cudaLaunchHostFunc(stream1, fn, &hostFnData);
    cudaStreamEndCapture(stream1, &graph);

    // Instantiate and launch the captured graph
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, streamForGraph);
    cudaStreamSynchronize(streamForGraph);

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(streamForGraph);
    cudaEventDestroy(forkStreamEvent);
    cudaEventDestroy(memsetEvent1);
    cudaEventDestroy(memsetEvent2);
}

// ============================================================
// main
// ============================================================
int main() {
    size_t inputSize   = 1 << 20;          // 1M floats
    size_t numOfBlocks = 256;

    float  *inputVec_h, *inputVec_d;
    double *outputVec_d, *result_d;

    // Allocate host & device memory
    inputVec_h = (float  *)malloc(inputSize * sizeof(float));
    cudaMalloc(&inputVec_d,  inputSize          * sizeof(float));
    cudaMalloc(&outputVec_d, numOfBlocks        * sizeof(double));
    cudaMalloc(&result_d,    1                  * sizeof(double));

    // Fill input on host: [1, 2, 3, ..., inputSize]
    for (size_t i = 0; i < inputSize; i++)
        inputVec_h[i] = (float)(i + 1);

    // Expected: sum 1..N = N*(N+1)/2
    double expected = (double)inputSize * (inputSize + 1) / 2.0;
    printf("Input size: %zu elements\n", inputSize);
    printf("Expected sum: %.0f\n", expected);

    cudaGraphsUsingStreamCapture(inputVec_h, inputVec_d,
                                 outputVec_d, result_d,
                                 inputSize, numOfBlocks);

    free(inputVec_h);
    cudaFree(inputVec_d);
    cudaFree(outputVec_d);
    cudaFree(result_d);
    return 0;
}
