#include <stdio.h>
#include <stdlib.h>
#include "io/matrixio.h"

int main(int argc, char *argv[])
{
    Matrix matrix;

    if(argc < 2){
        printf("Usage: exe FILE\n");
        printf("Error: Filename containing matrix not provided.\n");
        exit(-1);
    }

    matrix = buildArrayFromFile(argv[1]);
    printMatrix(matrix);
    printMatrixToFile(matrix, "matrix_result.txt");


    return 0;
}

