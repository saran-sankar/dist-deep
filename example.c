#include <stdio.h>
#include <layers.h>
#include <estimator.h>

int main(int argc, char *argv[])
{
    
    /*Load data*/
    
    int num_features = 4, num_samples = 150;
    int batch_size = 10;
    
    float* X = malloc(num_samples*num_features*sizeof(float));
    int* Y = malloc(num_samples*sizeof(float));
    float f1, f2, f3, f4;
    int c;
    
    FILE *file;
    file = fopen("iris.txt", "r");
    for (int i=0;i<num_samples/batch_size;i++){
        for (int j=0; j<batch_size; j++){
            
            fscanf(file, "%f,%f,%f,%f,%d", &f1, &f2, &f3, &f4, &c);

            X[i*num_features*batch_size + j+0] = f1;
            X[i*num_features*batch_size + j+batch_size] = f2;
            X[i*num_features*batch_size + j+2*batch_size] = f3;
            X[i*num_features*batch_size + j+3*batch_size] = f4;
            
            Y[i*batch_size + j] = c;
            
        }
    }
    
    /*Model*/
    
    struct model model;
    int num_layers = 3; /*input layer not counted*/
    struct layer input, dense1, dense2, output;
    
    
    /*Configure the model*/
    
    model.layers = (struct layer*) malloc((num_layers+1) * sizeof(struct layer));
    model.num_layers = num_layers;
    
    /*Configure input layer*/
    
    input.num_nodes = num_features;
    input.A = (float*) malloc(sizeof(X));
    input.A = X;
    
    model.num_features = num_features;
    model.input = input.A;

    /*Add layers to the model*/
    
    dense1 = Dense(input, 5);
    model.layers[0] = dense1;
    dense2 = Dense(dense1, 6);
    model.layers[1] = dense2;
    output = Dense(dense2, 3);
    model.layers[2] = output;
    
    /*Train the model*/

    int epochs = 5;
    
    model = DDClassifier(model, Y, num_samples, batch_size, epochs);
    
    return 0;
}
