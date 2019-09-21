//
//  layers.h
//  

#ifndef layers_h
#define layers_h

#include<omp.h>
#include <stdbool.h>
#include<stdlib.h>

struct layer{
    int num_nodes;
    
    float** W;
    float** b;
    float** Z;
    float** A;
    
    float** dW;
    float** db;
    float** dZ;
};

struct model{
    struct layer* Layers;
};

struct layer Dense(struct layer prevLayer, int num_nodes){
    
    struct layer denseLayer;
    int rows;
    int cols;
    
    denseLayer.num_nodes = num_nodes;
    
    rows = num_nodes;
    cols = prevLayer.num_nodes;
    
    /*Initilize the weights, bias, Z, activation matrix and and updation matrices*/
    
    denseLayer.W = (float**) malloc(rows * cols * sizeof(float));
    //denseLayer.b =
    //denseLayer.Z =
    //denseLayer.A =
    denseLayer.dW = (float**) malloc(rows * cols * sizeof(float));
    //denseLayer.db =
    //denseLayer.dZ =
    
    #pragma omp parallel for
    for (int i=0;i<rows;i++){
        denseLayer.W[i] = (float*) malloc(cols * sizeof(float));
        for (int j=0;j<cols;j++){
            denseLayer.W[i][j] = (rand() % 100)/100.0;
        }
    }
    
    return denseLayer;
}

#endif /* layers_h */