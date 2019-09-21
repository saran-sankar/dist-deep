#include <stdio.h>
#include <layers.h>


/*training data*/
float X[2][10] = {{1.3, 5.4, 6.4, 7.6, 6.4, 3.5, 6.5, 5.4, 2.8, 7.1},
    {1.3, 5.4, 6.4, 7.6, 6.4, 3.5, 6.5, 5.4, 2.8, 7.1}};
float Y[1] = {1};

int main(int argc, char *argv[])
{
    struct model Model;
    int num_features, num_layers = 3;
    struct layer input, dense1, dense2;
    
    /*Configure the model*/
    
    Model.Layers = (struct layer*) malloc(num_layers * sizeof(struct layer));
    
    /*Configure input layer*/
    
    num_features = (sizeof(X)) / (sizeof(X[0])); /*rows in input*/
    input.num_nodes = num_features;
    input.A = (float**) malloc(sizeof(X));
    
    #pragma omp parallel for
    for (int i=0;i<num_features;i++){
        input.A[i] = (float*) malloc(sizeof(X[0])/sizeof(float));
        input.A[i] = X[i];
    }

    /*Add layers to the model*/
    
    dense1 = Dense(input, 5);
    Model.Layers[0] = dense1;
    dense2 = Dense(dense1, 6);
    Model.Layers[1] = dense2;

    printf("%d\n", Model.Layers[0].num_nodes);
    //printf("%f \n", dense1.W[0][0]);
    
    for (int i=0;i<5;i++){
        for (int j=0;j<2;j++){
            printf("%f ", Model.Layers[0].W[i][j]);
        }
        printf("\n");
    }
    return 0;
}
