#include <stdio.h>
#include <stdlib.h>
#include "../include/matrixio.h"
#include "../include/cpddp.h"

//TODO: change dimensions from int to llu
int main(int argc, char *argv[])
{
    if(argc < 2){
        printf("Usage: exe FILE\n");
        printf("Error: Filename containing M not provided.\n");
        exit(-1);
    }

    Matrix M = buildMatrixFromFile(argv[1]);
    printMatrix(M);

    Matrix eigenvalue = pddpStep(M);
    printMatrix(eigenvalue);
    printMatrixToFile(eigenvalue, "result.mat");

    free(M.matrix);
    free(eigenvalue.matrix);
    return 0;
}
