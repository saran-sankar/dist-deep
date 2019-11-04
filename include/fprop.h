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
    
    float* sub_W = malloc(cols * sizeof(float));  /*sizeof(W[0])*/
    
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
    
    float* Z = malloc(cols * batch_size * sizeof(float));
    
    MPI_Allreduce(
                  sub_Z,
                  Z,
                  cols * batch_size,
                  MPI_FLOAT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    float* sub_Z2 = malloc(batch_size * sizeof(float));
    float* sub_A = malloc(batch_size * sizeof(float));
    
    
    if (rank < cols){
        
        for (int j=0; j<batch_size; j++){
            sub_Z2[j] = Z[rank*batch_size + j] + b[rank];
            sub_A[j] = 1.0/(1+exp(-sub_Z2[j]));
        }
            
    }
            
    else{
        
        for (int j=0; j<batch_size; j++){
            sub_Z2[j] = 0;
            sub_A[j] = 0;
        }
        
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    float* Z_new = malloc(cols * batch_size * sizeof(float));
    float* A = malloc(cols * batch_size * sizeof(float));
    
    if (rank == 2 && verbose>1){
        printf("Calculating A..\n");
    }
    
    MPI_Allgather(
                  sub_Z2,
                  batch_size,
                  MPI_FLOAT,
                  Z_new,
                  batch_size,
                  MPI_FLOAT,
                  MPI_COMM_WORLD
                  );
    
    if (rank == 2 && verbose>1){
        printf("Gathered Z\n");
    }
    
    MPI_Allgather(
                  sub_A,
                  batch_size,
                  MPI_FLOAT,
                  A,
                  batch_size,
                  MPI_FLOAT,
                  MPI_COMM_WORLD
                  );
    
    if (rank == 2 && verbose>1){
        printf("Gathered A\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    struct output output;
    
    output.Z = Z_new;
    output.A = A;
    
    return output;
}

#endif /* fprop_h */
