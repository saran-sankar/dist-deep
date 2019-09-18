//
//  layers.h
//  

#ifndef layers_h
#define layers_h

#include<mpi.h>
#include<omp.h>
#include <stdbool.h>
#include<stdlib.h>

struct layer{
    int num_nodes;
    float** W;
    float** b;
    float** Z;
    float** A;
};

struct model{
    struct layer* Layers;
};

struct layer Dense(struct layer prevLayer, int num_nodes){
    
    struct layer denseLayer;
    float** W;
    int rows;
    int cols;
    
    rows = num_nodes;
    cols = prevLayer.num_nodes;
    
    W = (float**) malloc(rows * sizeof(float));
        
    for (int i=0;i<rows;i++){
        W[i] = (float*) malloc(cols * sizeof(float));
        for (int j=0;j<cols;j++){
            W[i][j] = (rand() % 100)/100.0;
        }
    }
    
    denseLayer.W = (float**) malloc(2 * sizeof(float));
    denseLayer.W = W;
    
    return denseLayer;
}

#endif /* layers_h */
