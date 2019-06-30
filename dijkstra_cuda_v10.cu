//compile
/** nvcc -arch=sm_11 dijkstra_cuda.cu -o dijkstra_cuda **/

//reference: github.com/AlexDWong/dijkstra-CUDA/

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>


#define VERTICES 16384         //number of vertices
#define CPU_IMP 1               //number of Dijkstra implementations (non-GPU)
#define GPU_IMP 1               //number of Dijkstra implementations (GPU)
#define BLOCKS 32
#define THREADS_PER_BLOCK 512

typedef int data_t;


int main() {
	clock_t cpu_startTime, cpu_endTime;
	double cpu_ElapseTime = 0;

	int* d_min;
	int* d_minIndex;
	int* d_temp;
	int* d_tempIndex;

	cudaMalloc((void**)& d_min, sizeof(int));
	cudaMalloc((void**)& d_minIndex, sizeof(int));
	cudaMalloc((void**)& d_temp, BLOCKS * sizeof(int));
	cudaMalloc((void**)& d_tempIndex, BLOCKS * sizeof(int));

	cudaMemset(d_min, INT_MAX, sizeof(int));
	cudaMemset(d_minIndex, 0, sizeof(int));


	//functions
	void setIntArrayValue(int* in_array, int array_size, int value);
	void setDataArrayValue(data_t * in_array, int array_size, data_t init_value);
	void initializeGraphZero(data_t * graph, int num_vertices);
	void constructGraphEdge(data_t * graph, int num_vertices);

	//Dijkstra's implementations
	void dijkstraCPUSerial(data_t * graph, data_t * node_dist, int* visited_node, int num_vertices, int v_start);   //serial Dijkstra
	__global__ void closestNodeCUDA(int* d_min, int* d_minIndex, int* d_temp, int* d_tempIndex, data_t * node_dist, int* visited_node, int* global_closest, int num_vertices);                   //Dijkstra CUDA Pt. 1
	__global__ void cudaRelax(data_t * graph, data_t * node_dist, int* visited_node, int* source);                  //Dijkstra CUDA Pt. 2


	/*************SETUP GRAPH*************/
	int graph_size = VERTICES * VERTICES * sizeof(data_t);
	int int_array = VERTICES * sizeof(int);
	int data_array = VERTICES * sizeof(data_t);
	data_t* graph = (data_t*)malloc(graph_size);
	data_t* node_dist = (data_t*)malloc(data_array);
	int* visited_node = (int*)malloc(int_array);
	data_t* dist_matrix = (data_t*)malloc((CPU_IMP + GPU_IMP) * data_array);

	printf("Variables created, allocated\n");

	data_t* gpu_graph;      //CUDA mallocs
	data_t* gpu_node_dist;
	int* gpu_visited_node;
	cudaMalloc((void**)& gpu_graph, graph_size);
	cudaMalloc((void**)& gpu_node_dist, data_array);
	cudaMalloc((void**)& gpu_visited_node, int_array);
	int* closest_vertex = (int*)malloc(sizeof(int));    //for closest vertex
	int* gpu_closest_vertex;
	closest_vertex[0] = -1;
	cudaMalloc((void**)& gpu_closest_vertex, (sizeof(int)));
	cudaMemcpy(gpu_closest_vertex, closest_vertex, sizeof(int), cudaMemcpyHostToDevice);

	setDataArrayValue(node_dist, VERTICES, INT_MAX);
	setIntArrayValue(visited_node, VERTICES, 0);
	initializeGraphZero(graph, VERTICES);
	constructGraphEdge(graph, VERTICES);

	printf("Variables initialized.\n");

	/************RUN DIJKSTRA'S************/
	int i;
	int origin = 0;
	printf("Origin vertex: %d\n", origin);

	/*  SERIAL DIJKSTRA  */
	int version = 0;
	printf("Running serial...");
	cpu_startTime = clock();
	dijkstraCPUSerial(graph, node_dist, visited_node, VERTICES, origin);
	cpu_endTime = clock();
	cpu_ElapseTime = ((cpu_endTime - cpu_startTime) / (double)CLOCKS_PER_SEC);

	for (i = 0; i < VERTICES; i++) {
		dist_matrix[version * VERTICES + i] = node_dist[i];
	}
	printf("Done!\n");

	/*  CUDA DIJKSTRA  */
	version++;
	cudaEvent_t gpu_start, gpu_stop;
	float gpu_elapsed_exec;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);

	setDataArrayValue(node_dist, VERTICES, INT_MAX);   //reset previous data
	setIntArrayValue(visited_node, VERTICES, 0);
	node_dist[origin] = 0;

	cudaMemcpy(gpu_graph, graph, graph_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_node_dist, node_dist, data_array, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_visited_node, visited_node, int_array, cudaMemcpyHostToDevice);

	dim3 gridMin(BLOCKS, 1, 1);
	dim3 blockMin(THREADS_PER_BLOCK, 1, 1);

	dim3 gridRelax(BLOCKS, 1, 1); /** number of blocks **/
	dim3 blockRelax(THREADS_PER_BLOCK, 1, 1); /** numebr of threads per block **/


    cudaEventRecord(gpu_start, 0);
	for (int i = 0; i < VERTICES; i++) {
		closestNodeCUDA <<<gridMin, blockMin >>> (d_min, d_minIndex, d_temp, d_tempIndex, gpu_node_dist, gpu_visited_node, gpu_closest_vertex, VERTICES);  //find min
		cudaRelax <<<gridRelax, blockRelax >>> (gpu_graph, gpu_node_dist, gpu_visited_node, gpu_closest_vertex); //relax

	}
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_exec, gpu_start, gpu_stop);        //elapsed execution time
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

	cudaMemcpy(node_dist, gpu_node_dist, data_array, cudaMemcpyDeviceToHost);
	cudaMemcpy(visited_node, gpu_visited_node, int_array, cudaMemcpyDeviceToHost);
	for (i = 0; i < VERTICES; i++) {
		dist_matrix[version * VERTICES + i] = node_dist[i];
	}

	//free memory
	cudaFree(d_min);
	cudaFree(d_minIndex);
	cudaFree(d_temp);
	cudaFree(d_tempIndex);

	cudaFree(gpu_graph);
	cudaFree(gpu_node_dist);
	cudaFree(gpu_visited_node);


	printf("\nVertices: %d", VERTICES);
	printf("\n\nSerial Time (s): %.6f\n", cpu_ElapseTime);
	printf("\n\nCUDA Time (ms): %.3f\n", gpu_elapsed_exec);

	/***************ERROR CHECKING***************/
	printf("\n\nError checking:\n");
	printf("----Serial vs CUDA:\n");

	int d_errors = 0;
	for (i = 0; i < VERTICES; i++) {
		if (dist_matrix[i] != dist_matrix[VERTICES + i]) {
			d_errors++;
			printf("d_Error: Serial has %d %d, CUDA has %d %d\n", dist_matrix[i], i, dist_matrix[VERTICES + i], VERTICES + i);
		}
	}
	printf("--------%d dist errors found.\n", d_errors);
}


/****************DIJKSTRA'S ALGORITHM IMPLEMENTATIONS****************/
/* Serial Implementation */
int closestNode(data_t* node_dist, int* visited_node, int num_vertices) {
	int dist = INT_MAX;
	int node = -1;
	int i;

	for (i = 0; i < num_vertices; i++) {
		if ((node_dist[i] < dist) && (visited_node[i] == 0)) {
			node = i;
			dist = node_dist[i];
		}
	}
	return node;
}

void dijkstraCPUSerial(data_t* graph, data_t* node_dist, int* visited_node, int num_vertices, int v_start) {

	//functions
	void setIntArrayValue(int* in_array, int array_size, int init_value);
	void setDataArrayValue(data_t * in_array, int array_size, data_t init_value);
	int closestNode(data_t * node_dist, int* visited_node, int num_vertices);

	setDataArrayValue(node_dist, VERTICES, INT_MAX);   //reset data from previous runs
	setIntArrayValue(visited_node, VERTICES, 0);
	node_dist[v_start] = 0;

	int i, next;
	for (i = 0; i < num_vertices; i++) {
		int curr_node = closestNode(node_dist, visited_node, num_vertices);
		visited_node[curr_node] = 1;
		for (next = 0; next < num_vertices; next++) {   //Update only if neighbor is reachable, not visited, and if distance through current is less than the current min
			int new_dist = node_dist[curr_node] + graph[curr_node * num_vertices + next];
			if ((visited_node[next] != 1)
				&& (graph[curr_node * num_vertices + next] != (data_t)(0))
				&& (new_dist < node_dist[next])) {
				node_dist[next] = new_dist;        //update distance
			}
		}
	}
}

/* CUDA implementation */
__global__ void closestNodeCUDA(int* min_value, int* minIndex, int* temp, int* tempIndex, data_t* node_dist, int* visited_node, int* global_closest, int num_vertices) {

	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int cache[THREADS_PER_BLOCK];
	__shared__ int cacheIndex[THREADS_PER_BLOCK];


	if (index < num_vertices) {
		if ((node_dist[index]) < INT_MAX && (visited_node[index]) == 0) {
			cache[threadIdx.x] = node_dist[index];
			cacheIndex[threadIdx.x] = index;
		}
		else {
			cache[threadIdx.x] = INT_MAX;
			cacheIndex[threadIdx.x] = -1;
		}
	}
	__syncthreads();

	unsigned int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			if (cache[threadIdx.x + i] < cache[threadIdx.x]) {
				cache[threadIdx.x] = cache[threadIdx.x + i];
				cacheIndex[threadIdx.x] = cacheIndex[threadIdx.x + i];
			}
		}
		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		temp[blockIdx.x] = cache[0];
		tempIndex[blockIdx.x] = cacheIndex[0];
	}

	unsigned int k = BLOCKS / 2;
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		while (k != 0) {
			for (int j = 0; j < k; ++j) {
				if ((temp[j + k]) < temp[j]) {
					temp[j] = temp[j + k];
					tempIndex[j] = tempIndex[j + k];
				}
			}

			__syncthreads();
			k /= 2;
		}
	}
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*min_value = temp[0];
		*minIndex = tempIndex[0];

		global_closest[0] = *minIndex;
		visited_node[*minIndex] = 1;
	}
	__syncthreads();

}

__global__ void cudaRelax(data_t* graph, data_t* node_dist, int* visited_node, int* global_closest) {
	int next = blockIdx.x * blockDim.x + threadIdx.x;
	int source = global_closest[0];

	data_t edge = graph[source * VERTICES + next];
	data_t new_dist = node_dist[source] + edge;

	if ((edge != 0) &&
		(visited_node[next] != 1) &&
		(new_dist < node_dist[next])) {
		node_dist[next] = new_dist;
	}

}

/********FUNCTIONS*********/

/*  Initialize elements of a 1D int array with an initial value   */
void setIntArrayValue(int* in_array, int array_size, int init_value) {
	int i;
	for (i = 0; i < array_size; i++) {
		in_array[i] = init_value;
	}
}

/*  Initialize elements of a 1D data_t array with an initial value   */
void setDataArrayValue(data_t* in_array, int array_size, data_t init_value) {
	int i;
	for (i = 0; i < array_size; i++) {
		in_array[i] = init_value;
	}
}

/*  Construct graph with no edges or weights     */
void initializeGraphZero(data_t* graph, int num_vertices) {
	int i, j;

	for (i = 0; i < num_vertices; i++) {
		for (j = 0; j < num_vertices; j++) {           //weight of all edges initialized to 0
			graph[i * num_vertices + j] = (data_t)0;
		}
	}
}

/*  Construct a fully connected, undirected graph with non-negative edges and a minimum degree for vertices.  */
void constructGraphEdge(data_t* graph, int num_vertices) {
	int i, j;
	data_t weight;

	//initialize a connected graph
	printf("Initializing a connected graph...");
	for (i = 0; i < num_vertices; i++) {
		for (j = i; j < num_vertices; j++) {
			weight = rand() % 1000;
			graph[i * num_vertices + j] = weight;
			graph[j * num_vertices + i] = weight;
		}

	}
	printf("done!\n");
}
