#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Matrix{
    int rows, cols;
    double* M;
};

typedef struct Matrix Matrix;

Matrix buildArrayFromFile(char* filename);

int main(int argc, char *argv[])
{
    Matrix matrix;

    if(argc < 2){
        printf("Usage: exe FILE\n");
        printf("Error: Filename containing matrix not provided.\n");
        exit(-1);
    }

    matrix = buildArrayFromFile(argv[1]);

    printf("\n\n x: %d\n y: %d\n\n", matrix.rows, matrix.cols);
    for(int i=0; i<matrix.rows; i++){
        for(int j=0; j<matrix.cols; j++){
            printf("%0.3f  ", matrix.M[i*matrix.cols + j]);
        }
        printf("\n");
    }



    return 0;
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

    while ((c = fgetc(fp)) != EOF ){
        if (c == '\t') y++;
        else if(c == '\n') break;
    }
    while ((c = fgetc(fp)) != EOF ){
      if (c == '\n')
        x++;
    }

    matrix.rows = x;
    matrix.cols = y;
    matrix.M = malloc(y * x * sizeof(double));

    rewind(fp);

    char buffer[30];
    int i=0, buf_i=0;
    while ((c = fgetc(fp)) != EOF ){
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
