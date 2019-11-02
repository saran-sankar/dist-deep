//
//  estimator.h
//  

#ifndef estimator_h
#define estimator_h

#include<omp.h>
#include<mpi.h>
#include<math.h>

float* FProp(float * input, struct layer layer, int batch_size, int rank){
    
    float* W = layer.W;
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
    
    #pragma omp parallel for
    for (int i=0; i<cols * batch_size; i++){
        Z[i] = 1.0/(1+exp(-Z[i]));
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
                printf("%f ", Z[i*batch_size + j]);
            }
        }
    }
    
    float* output = Z;
    return output;
}

struct model DDClassifier(struct model model, int batch_size, int epochs){
    
    struct layer* layers;
    float* input;
    float* y_hat;
    int num_layers = model.num_layers;
    
    MPI_Init(NULL, NULL);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    input = model.input;
    
    for (int i=0; i<epochs; i++){
        
        if (rank==2){
            printf("\n\nEpoch %d, Forward propagation. Layer %d\n", i+1, 0+1);
        }
        
        model.layers[0].A = FProp(input, model.layers[0], batch_size, rank);
        
        /*Cannot be parallelized*/
        for (int j=1; j<num_layers; j++){
            if (rank==2){
                printf("\n\nEpoch %d, Forward propagation. Layer %d\n", i+1, j+1);
            }
            model.layers[j].A = FProp(model.layers[j-1].A, model.layers[j], batch_size, rank);
        }
        
        y_hat = model.layers[num_layers-1].A;
        
        /*Softmax*/
        
        float* y_hat_sums = malloc(batch_size * sizeof(float));
        for (int j=0; j<model.layers[num_layers-1].num_nodes;j++){
            for (int k=0; k<batch_size; k++){
                y_hat_sums[k] += y_hat[j*batch_size + k];
            }
        }
        for (int j=0; j<model.layers[num_layers-1].num_nodes;j++){
            for (int k=0; k<batch_size; k++){
                y_hat[j*batch_size + k] /= y_hat_sums[k];
            }
        }
        
        if (rank==2){
            printf("\n\nEpoch %d, Forward propagation. Softmax output\n", i+1);
            for (int j=0; j<model.layers[num_layers-1].num_nodes * batch_size;j++){
                printf("%f ", y_hat[j]);
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
                
    //printf("%f Status: OK\n", input[9]);
    printf("\n");
    return model;
}

#endif /* estimator_h */
