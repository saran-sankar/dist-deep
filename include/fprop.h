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

struct output FProp(float * input, struct layer layer, int batch_size, int rank, int verbose){
    
    float* W = layer.W;
    float* b = layer.b;
    int cols = layer.num_nodes;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    float* sub_Z = malloc(cols * batch_size * sizeof(float));
    
    /*Calculate Z*/
    
    if (rank < layer.prev_num_nodes){
        
        /*Display process details*/
        
        char processor_name [MPI_MAX_PROCESSOR_NAME];
        int nprocs, resultlen;
        
        MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
        
        MPI_Get_processor_name(processor_name, &resultlen);
        
        if (verbose>1){
            printf("Pocess %d of %d calculating Z on %s\n", rank, nprocs, processor_name);
            fflush(stdout);
        }
        
#pragma omp parallel for collapse(2)
        for (int i=0; i<cols; i++){
            for (int j=0; j<batch_size; j++){
                sub_Z[i*batch_size + j] = W[rank*cols + i] * input[rank*batch_size + j];
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
    
    if (rank == 2 && verbose>1){
        printf("All processes calculating A\n");
    }
    
#pragma omp parallel for collapse(2)
    for (int i=0; i<cols; i++){
        for (int j=0; j<batch_size; j++){
            A[i*batch_size + j] = 1.0/(1+exp(-Z[i*batch_size + j]));
        }
    }
    
    struct output output;
    
    output.Z = Z;
    output.A = A;
    
    return output;
}

#endif /* fprop_h */
