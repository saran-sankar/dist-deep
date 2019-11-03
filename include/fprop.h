//
//  fprop.h
//  

#ifndef fprop_h
#define fprop_h

#include<omp.h>
#include<mpi.h>
#include<math.h>

struct output{
    float* A;
    float* Z;
};

struct output FProp(float * input, struct layer layer, int batch_size, int rank){
    
    float* W = layer.W;
    float* b = layer.b;
    int cols = layer.num_nodes;
    
    float* sub_W = malloc(cols*sizeof(float));  /*sizeof(W[0])*/
    
    MPI_Scatter(
                W,
                cols,
                MPI_FLOAT,
                sub_W,
                cols,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);
    
    float* sub_input = malloc(batch_size * sizeof(float));
    
    MPI_Scatter(
                input,
                batch_size,
                MPI_FLOAT,
                sub_input,
                batch_size,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    float* sub_Z = malloc(cols * batch_size * sizeof(float));
    
    //printf("%d\n",layer.prev_num_nodes);
    
    if (rank < layer.prev_num_nodes){
#pragma omp parallel for collapse(2)
        for (int i=0; i<cols; i++){
            for (int j=0; j<batch_size; j++){
                sub_Z[i*batch_size + j] = sub_W[i] * sub_input[j];
            }
        }
    }
    else{
#pragma omp parallel for collapse(2)
        for (int i=0; i<cols; i++){
            for (int j=0; j<batch_size; j++){
                sub_Z[i*batch_size + j] = 0;
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    //printf("rank = %d\n", rank);
    
    float* Z = malloc(cols * batch_size * sizeof(float));
    
    MPI_Allreduce(
                  sub_Z,
                  Z,
                  cols * batch_size,
                  MPI_FLOAT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
#pragma omp parallel for collapse(2)
    for (int i=0; i<cols; i++){
        for (int j=0; j<batch_size; j++){
            Z[i*batch_size + j] = Z[i*batch_size + j] + b[i];
        }
    }
    
    float* A = malloc(cols * batch_size * sizeof(float));
    
#pragma omp parallel for collapse(2)
    for (int i=0; i<cols; i++){
        for (int j=0; j<batch_size; j++){
            A[i*batch_size + j] = 1.0/(1+exp(-Z[i*batch_size + j]));
        }
    }
    
    if (rank == 2){
        for (int i=0; i<cols; i++){
            for (int j=0; j<batch_size; j++){
                //printf("%f ", sub_Z[i*batch_size + j]);
            }
        }
    }
    
    if (rank == 2){
        for (int i=0; i<cols; i++){
            for (int j=0; j<batch_size; j++){
                //printf("%f ", A[i*batch_size + j]);
            }
        }
    }
    
    struct output output;
    
    output.Z = Z;
    output.A = A;
    
    return output;
}

#endif /* fprop_h */
