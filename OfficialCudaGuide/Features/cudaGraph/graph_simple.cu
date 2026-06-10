#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernelName(int *data) {
    printf("  Kernel executed by thread %d\n", threadIdx.x);
}

int main() {
    int *d_data;
    cudaMalloc(&d_data, sizeof(int));

    // Create the graph - it starts out empty
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    // Create the nodes and their dependencies
    cudaGraphNode_t nodes[4];
    cudaGraphNodeParams kParams = { cudaGraphNodeTypeKernel };
    kParams.kernel.func         = (void *)kernelName;
    kParams.kernel.gridDim.x    = kParams.kernel.gridDim.y  = kParams.kernel.gridDim.z  = 1;
    kParams.kernel.blockDim.x   = kParams.kernel.blockDim.y = kParams.kernel.blockDim.z = 1;
    kParams.kernel.kernelParams = (void *[]){&d_data};

    cudaGraphAddNode(&nodes[0], graph, NULL, NULL, 0, &kParams);
    cudaGraphAddNode(&nodes[1], graph, &nodes[0], NULL, 1, &kParams);
    cudaGraphAddNode(&nodes[2], graph, &nodes[0], NULL, 1, &kParams);
    cudaGraphAddNode(&nodes[3], graph, &nodes[1], NULL, 1, &kParams);

    printf("Dependency: Node 0 -> Node 1 -> Node 3\n");
    printf("            Node 0 -> Node 2\n\n");

    // Instantiate, launch, cleanup
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    printf("\nGraph execution complete!\n");

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_data);
    return 0;
}
