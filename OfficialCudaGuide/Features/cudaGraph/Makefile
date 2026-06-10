NVCC = nvcc
NVCC_FLAGS = -arch=native

.PHONY: all clean

all: graph_simple.cu graph_stream_capture.cu

graph_simple.cu:
	$(NVCC) $(NVCC_FLAGS) -o graph_simple graph_simple.cu

graph_stream_capture.cu:
	$(NVCC) $(NVCC_FLAGS) -o graph_stream_capture graph_stream_capture.cu

clean:
	rm -f graph_simple graph_stream_capture
