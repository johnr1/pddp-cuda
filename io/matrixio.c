#include <stdio.h>
#include <stdlib.h>
#include "matrixio.h"

void printMatrix(Matrix matrix){
    printf("Matrix [%dx%d] = \n", matrix.rows, matrix.cols);
    for(int i=0; i<matrix.rows; i++){
        for(int j=0; j<matrix.cols; j++){
            printf("\t%0.3f  ", matrix.M[i*matrix.cols + j]);
        }
        printf("\n");
    }
}

void printMatrixToFile(Matrix matrix, char* fileName){
    FILE *fp = fopen(fileName, "w");
    for(int i=0; i<matrix.rows; i++){
        fprintf(fp,"%f", matrix.M[i*matrix.cols]);
        for(int j=1; j<matrix.cols; j++){
            fprintf(fp,"\t%.15f", matrix.M[i*matrix.cols + j]);
        }
        fprintf(fp ,"\n");
    }
}

Matrix buildArrayFromFile(char* filename){
    Matrix matrix;
    FILE *fp = fopen(filename, "r");
    int x = 1, y = 1;
    char c;

    if(fp == NULL){
        printf("File not found, exiting.\n");
        exit(-2);
    }

    while ((c = (char)fgetc(fp)) != EOF ){
        if (c == '\t') y++;
        else if(c == '\n') break;
    }
    while ((c = (char)fgetc(fp)) != EOF ){
        if (c == '\n')
            x++;
    }

    matrix.rows = x;
    matrix.cols = y;
    matrix.M = malloc(y * x * sizeof(double));

    rewind(fp);

    char buffer[30];
    int i=0, buf_i=0;
    while ((c = (char)fgetc(fp)) != EOF ){
        if (c == '\t' || c == '\n')
        {
            buffer[buf_i] = '\0';
            matrix.M[i] = atof(buffer);
            buf_i = 0;
            i++;
        }
        else{
            buffer[buf_i++] = c;
        }
    }

    fclose(fp);

    return matrix;
}
