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
    
    float* W;
    float* b;
    float* Z;
    float* A;
    
    float* dW;
    float* db;
    float* dZ;
};

struct model{
    int num_layers;
    float * input;
    struct layer* layers;
};

struct layer Dense(struct layer prevLayer, int num_nodes){
    
    struct layer denseLayer;
    int rows;
    int cols;
    
    denseLayer.num_nodes = num_nodes;
    
    rows = prevLayer.num_nodes;
    cols = num_nodes;
    
    /*Initilize the weights, bias, Z, activation matrix and and updation matrices*/
    
    denseLayer.W = (float*) malloc(rows * cols * sizeof(float));
    //denseLayer.b =
    //denseLayer.Z =
    //denseLayer.A =
    denseLayer.dW = (float*) malloc(rows * cols * sizeof(float));
    //denseLayer.db =
    //denseLayer.dZ =
    
    #pragma omp parallel for
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            denseLayer.W[i*rows+j] = (rand() % 100)/100.0;
        }
    }
    
    return denseLayer;
}

#endif /* layers_h */
