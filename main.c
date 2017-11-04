#include <stdio.h>
#include <stdlib.h>
#include "io/matrixio.h"
#include "cpddp/cpddp.h"

int main(int argc, char *argv[])
{
    if(argc < 2){
        printf("Usage: exe FILE\n");
        printf("Error: Filename containing M not provided.\n");
        exit(-1);
    }

    Matrix M = buildArrayFromFile(argv[1]);
    printMatrix(M);

    Matrix eigenvalue = pddpStep(M);
    printMatrix(eigenvalue);


    free(M.matrix);
    free(eigenvalue.matrix);
    return 0;
}
