//
//  estimator.h
//  

#ifndef estimator_h
#define estimator_h

#include<omp.h>
#include<mpi.h>
#include<math.h>
#include<fprop.h>
#include<bprop.h>

struct model DDClassifier(struct model model, int* Y, int num_samples, int batch_size, int epochs, float learning_rate){
    
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
    
    struct output output;
    
    /*Train model*/
    for (int i=0; i<epochs; i++){
        
        float epoch_loss = 0.0;
        
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
                printf("\n\nEpoch %d, Batch %d \n\nForward propagation. Layer %d\n", i+1, batch+1, 0+1);
            }
            
            output = FProp(input, model.layers[0], batch_size, rank);
            model.layers[0].Z = output.Z;
            model.layers[0].A = output.A;
            
            /*Cannot be parallelized*/
            for (int j=1; j<num_layers; j++){
                if (rank==2){
                    printf("\n\nEpoch %d, Batch %d \n\nForward propagation. Layer %d\n", i+1, batch+1, j+1);
                }
                
                output = FProp(model.layers[j-1].A, model.layers[j], batch_size, rank);
                model.layers[j].Z = output.Z;
                model.layers[j].A = output.A;
            }
            
            y_hat = model.layers[num_layers-1].A;
            
            int num_classes = model.layers[num_layers-1].num_nodes;
            
            if (rank==2){
                if (rank==2){
                    printf("\nCalculating output..\n");
                }
                //printf("\n\nEpoch %d, Forward propagation. Output\n", i+1);
                for (int j=0; j<num_classes * batch_size;j++){
                    //printf("%f ", y_hat[j]);
                }
            }
            
            /*Loss*/
            
            if (rank==2){
                printf("Calculating loss..\n");
            }
            
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
            
            loss /= -batch_size*num_classes;
            epoch_loss += loss;
            
            model = BProp(model, y_hat, y, batch_size, learning_rate, rank);
            
        }
        
        epoch_loss /= num_batches;
        if (rank==2){
            printf("\n\n\n\n\n\nloss: %f\n\n\n\n\n\n", epoch_loss);
        }
        
        
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    //printf("%f Status: OK\n", input[9]);
    return model;
}

#endif /* estimator_h */
