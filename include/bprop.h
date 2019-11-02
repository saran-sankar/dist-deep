//
//  bprop.h
//  

#ifndef bprop_h
#define bprop_h

#include<omp.h>
#include<mpi.h>
#include<math.h>

struct model BProp(struct model model, float* y_hat, int* y, int batch_size, float learning_rate){
    
    int num_layers = model.num_layers;
    int num_classes = model.layers[num_layers-1].num_nodes;
    
    float* dZ_loss;
    dZ_loss = malloc(num_classes*batch_size*sizeof(float));
    for (int j=0; j<num_classes; j++){
        for (int k=0; k<batch_size; k++){
            if(j==y[k]){
                dZ_loss[j*num_classes + k] = y_hat[j*num_classes + k] - 1;
            }
            else{
                dZ_loss[j*num_classes + k] = y_hat[j*num_classes + k];
            }
        }
    }
    model.layers[num_layers-1].dZ = dZ_loss;
    
    float* A;
    float* dZ;
    float* dW;
    float* W;
    
    for (int i=1; i<2/*num_layers*/; i++){
        A = model.layers[num_layers-(i+1)].A; //n[i-1] * m
        dZ = model.layers[num_layers-i].dZ; //n[i] * m
        dW = model.layers[num_layers-i].dW; //n[i-1] * n[i]
        
        int num_nodes = model.layers[num_layers-i].num_nodes;
        int num_nodes_prev = model.layers[num_layers-(i+1)].num_nodes;
        
        #pragma omp parallel for collapse(2)
        for (int j=0; j<num_nodes_prev; j++){
            for (int k=0; k<num_nodes; k++){
                dW[j*num_nodes_prev + k] = 0.0;
                for (int m=0; m<batch_size; m++){
                    dW[j*num_nodes_prev + k] += A[j*num_nodes_prev + m] * dZ[k*num_nodes + m];
                }
            }
        }
        
        W = model.layers[num_layers-i].W;
        
        for (int j=0; j<num_nodes*num_nodes_prev; j++){
            W[j] -= learning_rate * dW[j];
        }
        
        model.layers[num_layers-i].W = W;
        
    }
    
    return model;
}

#endif /* bprop_h */
