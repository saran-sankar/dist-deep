//
//  estimator.h
//  

#ifndef estimator_h
#define estimator_h

#include<omp.h>
#include<mpi.h>

float * FProp(float * input, struct layer layer, int rank){
    
    float *W = layer.W;
    int cols = layer.num_nodes;
    
    float *sub_W = malloc(cols*sizeof(float));  /*sizeof(W[0])*/

    MPI_Scatter(
                W,
                cols,
                MPI_FLOAT,
                sub_W,
                cols,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);
    
    
    
    MPI_Reduce(
               void* send_data,
               void* recv_data,
               int count,
               MPI_Datatype datatype,
               MPI_Op op,
               int root,
               MPI_Comm communicator);
    
    
    float * output = input;
    return output;
}

struct model DDClassifier(struct model model, int batch_size, int epochs){
    
    struct layer * layers;
    float * input;
    float * y_hat;
    int num_layers = model.num_layers;
    
    MPI_Init(NULL, NULL);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    input = model.input;
    
    for (int i=0; i<epochs; i++){
        
        model.layers[0].A = FProp(input, model.layers[0], rank);
        
        /*Cannot be parallelized*/
        for (int j=1; j<num_layers; j++){
            model.layers[j].A = FProp(model.layers[j-1].A, model.layers[j], rank);
        }
        
        y_hat = model.layers[num_layers-1].A;
        
    }
    printf("%f Status: OK\n", input[9]);
    return model;
}

#endif /* estimator_h */
