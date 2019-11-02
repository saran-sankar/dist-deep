//
//  estimator.h
//  

#ifndef estimator_h
#define estimator_h

#include<omp.h>
#include<mpi.h>
#include<math.h>
#include<fprop.h>

struct model DDClassifier(struct model model, int* Y, int num_samples, int batch_size, int epochs){
    
    struct layer* layers;
    float* model_input;
    float* y_hat;
    int num_layers = model.num_layers;
    
    int num_batches = num_samples/batch_size;
    int num_features = model.num_features;
    float* input = malloc(num_features * batch_size * sizeof(float));
    int* y = malloc(batch_size*sizeof(int));
    
    MPI_Init(NULL, NULL);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    model_input = model.input;
    
    for (int i=0; i<epochs; i++){
        
        /*Cannot be parallelized*/
        for (int batch=0; batch<num_batches; batch++){
            
            #pragma omp parallel for collapse(2)
            for(int j=0; j<num_features; j++){
                for(int k=0; k<batch_size; k++){
                    input[j*num_features + k] = model_input[batch*num_batches + j*num_features + k];
                }
            }
            
            #pragma omp parallel for
            for(int j=0; j<batch_size; j++){
                y[j] = Y[batch*batch_size + j];
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
            
            if (rank==2){
                //printf("\n\nEpoch %d, Forward propagation. Layer %d\n", i+1, 0+1);
            }
            
            model.layers[0].A = FProp(input, model.layers[0], batch_size, rank);
            
            /*Cannot be parallelized*/
            for (int j=1; j<num_layers; j++){
                if (rank==2){
                    //printf("\n\nEpoch %d, Forward propagation. Layer %d\n", i+1, j+1);
                }
                model.layers[j].A = FProp(model.layers[j-1].A, model.layers[j], batch_size, rank);
            }
            
            y_hat = model.layers[num_layers-1].A;
            
            /*Softmax*/
            
            int num_classes = model.layers[num_layers-1].num_nodes;
            
            float* y_hat_sums = malloc(batch_size * sizeof(float));
            for (int j=0; j<batch_size; j++){
                y_hat_sums[j] = 0;
            }
            #pragma omp parallel for collapse(2)
            for (int j=0; j<num_classes; j++){
                for (int k=0; k<batch_size; k++){
                    y_hat_sums[k] += y_hat[j*batch_size + k];
                }
            }
            #pragma omp parallel for collapse(2)
            for (int j=0; j<num_classes; j++){
                for (int k=0; k<batch_size; k++){
                    y_hat[j*batch_size + k] /= y_hat_sums[k];
                }
            }
            
            if (rank==2){
                printf("\n\nEpoch %d, Forward propagation. Softmax output\n", i+1);
                for (int j=0; j<num_classes * batch_size;j++){
                    printf("%f ", y_hat[j]);
                }
            }
            
            /*Loss*/
            
            float loss = 0.0;
            
            for (int j=0; j<num_classes; j++){
                for (int k=0; k<batch_size; k++){
                    if(j==y[k]){
                        loss += log(y_hat[j*num_classes + k]);
                    }
                    else{
                        loss += log(1 - y_hat[j*num_classes + k]);
                    }
                }
            }
            
            loss /= batch_size*num_classes;
            
            if (rank==2){
                printf("\n\nloss:%f\n", loss);
            }
            
        }
        
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    //printf("%f Status: OK\n", input[9]);
    printf("\n");
    return model;
}

#endif /* estimator_h */
