#include <iostream>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include "omp.h"
#define THREADS 8

using namespace std;

const int N = 16384;

//serial findMin
int findMin(int *dist, bool *visit){

    int min = INT_MAX;
    int minNode = 0;

    for(int i = 0; i < N; ++i){
        if(visit[i] == false && dist[i] < min){
            min = dist[i];
            minNode = i;
        }
    }
    return minNode;
}


//parallel finMin
int findMin_p(int *dist_p, bool *visit_p){
    int min = INT_MAX;
    int minNode = 0;
    int min_dist_thread, min_node_thread;

    int vertex;

    omp_set_num_threads(THREADS);
#pragma omp parallel private(min_dist_thread, min_node_thread) shared(dist_p, visit_p)
    {
        min_dist_thread = min;
        min_node_thread = minNode;
#pragma omp barrier

#pragma omp for nowait
        for (vertex = 0; vertex < N; vertex += 4) {
            if ((dist_p[vertex] < min_dist_thread) && (visit_p[vertex] == false)) {
                min_dist_thread = dist_p[vertex];
                min_node_thread = vertex;
            }
            if(vertex+1 < N){
                if ((dist_p[vertex+1] < min_dist_thread) && (visit_p[vertex+1] == false)) {
                    min_dist_thread = dist_p[vertex+1];
                    min_node_thread = vertex+1;
                }
            }
            if(vertex+2 < N){
                if ((dist_p[vertex+2] < min_dist_thread) && (visit_p[vertex+2] == false)) {
                    min_dist_thread = dist_p[vertex+2];
                    min_node_thread = vertex+2;
                }
            }
            if(vertex+3 < N){
                if ((dist_p[vertex+3] < min_dist_thread) && (visit_p[vertex+3] == false)) {
                    min_dist_thread = dist_p[vertex+3];
                    min_node_thread = vertex+3;
                }
            }
        }

#pragma omp critical
        {
            if (min_dist_thread < min) {
                min = min_dist_thread;
                minNode = min_node_thread;
            }
        }
    }
    return minNode;
}


void output(int *dist){
    printf("Vertex   Distance from Source\n");
    for(int c = 0; c < N; ++c){
        printf("%d \t\t %d \t %d\n", c, dist[c], omp_get_thread_num());
    }
}


//serial Dijkstra
void Dijkstra(int **graph, int source, int *dist, bool *visit){

    for(int i = 0; i < N; ++i){
        dist[i] = INT_MAX;
        visit[i] = false;
    }

    dist[source] = 0;

    for(int j = 0; j < N; ++j){
        int minNode = findMin(dist, visit);
        visit[minNode] = true;

        for(int v = 0; v < N; ++v){
            if(visit[v] == false && dist[minNode] != INT_MAX && graph[minNode][v] != 0 && dist[v] > dist[minNode] + graph[minNode][v]){
                dist[v] = dist[minNode] + graph[minNode][v];
            }
        }
    }
}


//parallel Dijkstra
void Dijkstra_p(int **graph, int source, int *dist_p, bool *visit_p){

    for(int i = 0; i < N; ++i){
        dist_p[i] = INT_MAX;
        visit_p[i] = false;
    }

    dist_p[source] = 0;

    for(int j = 0; j < N; ++j){
        int minNode = findMin_p(dist_p, visit_p);
        visit_p[minNode] = true;

        omp_set_num_threads(THREADS);
#pragma omp parallel
        {
#pragma omp for
            for(int v = 0; v < N; v += 4){
                if(visit_p[v] == false && dist_p[minNode] != INT_MAX && graph[minNode][v] != 0 && dist_p[v] > dist_p[minNode] + graph[minNode][v]){
                    dist_p[v] = dist_p[minNode] + graph[minNode][v];
                }
                if(v+1 < N){
                    if(visit_p[v+1] == false && dist_p[minNode] != INT_MAX && graph[minNode][v+1] != 0 && dist_p[v+1] > dist_p[minNode] + graph[minNode][v+1]){
                        dist_p[v+1] = dist_p[minNode] + graph[minNode][v+1];
                    }
                }
                if(v+2 < N){
                    if(visit_p[v+2] == false && dist_p[minNode] != INT_MAX && graph[minNode][v+2] != 0 && dist_p[v+2] > dist_p[minNode] + graph[minNode][v+2]){
                        dist_p[v+2] = dist_p[minNode] + graph[minNode][v+2];
                    }
                }
                if(v+3 < N){
                    if(visit_p[v+3] == false && dist_p[minNode] != INT_MAX && graph[minNode][v+3] != 0 && dist_p[v+3] > dist_p[minNode] + graph[minNode][v+3]){
                        dist_p[v+3] = dist_p[minNode] + graph[minNode][v+3];
                    }
                }
            }
#pragma omp barrier
        }

    }
}


int main(int argc, const char * argv[]) {
    double start_time = 0.0;
    double end_time = 0.0;
    double duration = 0.0;
    int **graph;
    int *dist;
    bool *visit;
    int *dist_p;
    bool *visit_p;

    //    serial
    dist = (int *)malloc(N*sizeof(int));
    visit = (bool *)malloc(N*sizeof(bool));

    graph = (int**)malloc(N*sizeof(int*));
    for(int i = 0; i < N; ++i){
        graph[i] = (int*)malloc(N*sizeof(int));
    }
    for(int i = 0; i < N; ++i){
        for(int j = i; j < N; ++j){
            graph[i][j] = rand()%1000;
            graph[j][i] = graph[i][j];
        }
    }


    start_time = omp_get_wtime();
    Dijkstra(graph, 0, dist, visit);
    end_time = omp_get_wtime();
    duration = end_time - start_time;
    cout << "serial running time:" << 1000 * duration << "ms\n";

    //    parallel
    dist_p = (int *)malloc(N*sizeof(int));
    visit_p = (bool *)malloc(N*sizeof(bool));

    start_time = omp_get_wtime();
    Dijkstra_p(graph, 0, dist_p, visit_p);
    free(graph);
    end_time = omp_get_wtime();
    duration = end_time - start_time;
    cout << "parallel running time:" << 1000 * duration << "ms\n";

    printf("\nError checking:\n");

    printf("----Serial vs OPenMP:\n");
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (dist[i] != dist_p[i]) {
            errors++;
        }
    }
    printf("--------%d errors found.\n", errors);

    free(dist);
    free(visit);
    free(dist_p);
    free(visit_p);

    return 0;
}
