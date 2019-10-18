#include <stdio.h>
#include <layers.h>
#include <estimator.h>

/*training data*/
float X[2][10] = {{1.3, 5.4, 6.4, 7.6, 6.4, 3.5, 6.5, 5.4, 2.8, 7.1},
    {1.4, 7.2, 3.4, 9.3, 3.1, 3.0, 6.1, 8.2, 2.5, 3.8}};
float Y[10] = {1,0,2,0,0,1,2,1,1,2};

int main(int argc, char *argv[])
{
    struct model model;
    int num_features, num_samples, num_layers = 3; /*input layer not counted*/
    struct layer input, dense1, dense2, output;
    
    /*Configure the model*/
    
    model.layers = (struct layer*) malloc((num_layers+1) * sizeof(struct layer));
    model.num_layers = num_layers;
    
    /*Configure input layer*/
    
    num_features = sizeof(X) / sizeof(X[0]); /*rows in input*/
    num_samples = sizeof(X[0]) / sizeof(float);
    input.num_nodes = num_features;
    input.A = (float*) malloc(sizeof(X));
    
    int k = 0;
    for (int i=0;i<num_samples;i++){
        for (int j=0;j<num_features;j++)
        input.A[k++] = X[j][i];
    }
    
    model.input = input.A;

    /*Add layers to the model*/
    
    dense1 = Dense(input, 5);
    model.layers[0] = dense1;
    dense2 = Dense(dense1, 6);
    model.layers[1] = dense2;
    output = Dense(dense2, 3);
    model.layers[2] = output;
    
    /*Train the model*/
    
    int m = 2;
    int epochs = 5;
    
    model = DDClassifier(model, m, epochs);
    
    return 0;
}
