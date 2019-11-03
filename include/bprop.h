//
//  bprop.h
//  

#ifndef bprop_h
#define bprop_h

#include<omp.h>
#include<mpi.h>
#include<math.h>

struct model BProp(struct model model, float* y_hat, int* y, int batch_size, float learning_rate, int rank, int verbose){
    
    int num_layers = model.num_layers;
    int num_classes = model.layers[num_layers-1].num_nodes;
    
    if (rank == 2 && verbose>0){
        printf("\nStarting Backpropagation..\n\n");
    }
    
    /*Find the derivative of Z of last layer with respect to loss (sigmoid activation)*/
    
    float* dZ_loss;
    dZ_loss = malloc(num_classes*batch_size*sizeof(float));
    for (int j=0; j<num_classes; j++){
        for (int k=0; k<batch_size; k++){
            if(j==y[k]){
                dZ_loss[j*batch_size + k] = y_hat[j*batch_size + k] - 1;
            }
            else{
                dZ_loss[j*batch_size + k] = y_hat[j*batch_size + k];
            }
        }
    }
    model.layers[num_layers-1].dZ = dZ_loss;
    
    float* A_prev;
    float* dZ;
    float* dW;
    float* db;
    float* W;
    float* b;
    float* A;
    float* Z;
    float* Z_b;
    float* dZZ_b;
    float* I_A_prev;
    float* dZ_prev;
    
    dZZ_b = malloc(batch_size*batch_size*sizeof(float));
    
    for (int i=1; i<=num_layers; i++){
        
        A_prev = model.layers[num_layers-(i+1)].A; // shape: (n[N-1], m)
        dZ = model.layers[num_layers-i].dZ; // shape: (n[N], m)
        dW = model.layers[num_layers-i].dW; // shape: (n[N-1], n[N])
        db = model.layers[num_layers-i].db; // shape: (n[N], 1)
        
        int num_nodes = model.layers[num_layers-i].num_nodes;
        int num_nodes_prev = model.layers[num_layers-(i+1)].num_nodes;
        
        /*Find the derivative of weight with respect to loss (sigmoid activations)*/
        
#pragma omp parallel for collapse(2)
        for (int j=0; j<num_nodes_prev; j++){
            for (int k=0; k<num_nodes; k++){
                dW[j*num_nodes + k] = 0.0;
                for (int m=0; m<batch_size; m++){
                    dW[j*num_nodes + k] += A_prev[j*batch_size + m] * dZ[k*batch_size + m];
                }
            }
        }
        
        /*Find the derivative of bias with respect to loss (sigmoid activations)*/
        
        for (int j=0; j<num_nodes; j++){
            db[j] = 0.0;
            for (int k=0; k<batch_size; k++){
                db[j] += dZ[j*batch_size + k] / batch_size;
            }
        }
        
        /*Update weight*/
        
        if (rank == 2 && verbose>1){
            printf("Backpropagation, Layer %d, Updating weight and bias\n", num_layers-i+1);
        }
        
        W = model.layers[num_layers-i].W;
        
        for (int j=0; j<num_nodes*num_nodes_prev; j++){
            W[j] -= learning_rate * dW[j];
        }
        
        model.layers[num_layers-i].W = W;
        
        /*Update bias*/
        
        b = model.layers[num_layers-i].b; // shape: (n[N], 1)
        
        for (int j=0; j<num_nodes; j++){
            b[j] -= learning_rate * db[j];
        }
        
        model.layers[num_layers-i].b = b;
        
        /*Find the derivative of Z of inner layers with respect to loss (sigmoid activations)*/
        
        if (i != num_layers){
            
            Z = model.layers[num_layers-i].Z; // shape: (n[N], m)
            
            Z_b = malloc(num_nodes*batch_size*sizeof(float)); // Z-b
            
#pragma omp parallel for collapse(2)
            for (int j=0; j<num_nodes; j++){
                for(int k=0; k<batch_size; k++){
                    Z_b[j*batch_size + k] = Z[j*batch_size + k] - b[j];
                }
            }
            
#pragma omp parallel for collapse(2)
            for (int j=0; j<batch_size; j++){
                for (int k=0; k<batch_size; k++){
                    dZZ_b[j*batch_size + k] = 0.0;
                    for (int n=0; n<num_nodes; n++){
                        //dZZ_b[j*batch_size + k] += Z_b[j + n*batch_size] * dZ[k + n*batch_size];
                        dZZ_b[j*batch_size + k] += dZ[j + n*batch_size] * Z_b[k + n*batch_size];
                    }
                }
            }
            
            A = model.layers[num_layers-i].A; // shape: (n[N], m)
            
            I_A_prev = malloc(num_nodes_prev*batch_size*sizeof(float));
            
            for(int j=0; j<num_nodes_prev*batch_size; j++){
                I_A_prev[j] = 1 - A_prev[j];
            }
            
            dZ_prev = model.layers[num_layers-(i+1)].dZ; // shape: (n[N-1], m)
            dZ_prev = malloc(num_nodes_prev*batch_size*sizeof(float));
            
#pragma omp parallel for collapse(2)
            for (int j=0; j<num_nodes_prev; j++){
                for (int k=0; k<batch_size; k++){
                    dZ_prev[j*batch_size + k] = 0.0;
                    for (int m=0; m<batch_size; m++){
                        dZ_prev[j*batch_size + k] += I_A_prev[j*batch_size + m] * dZZ_b[k*batch_size + m];
                    }
                }
            }
            
            model.layers[num_layers-(i+1)].dZ = dZ_prev;
            
        }
    }
    
    return model;
}

#endif /* bprop_h */
