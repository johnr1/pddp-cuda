#include <stdio.h>
#include <stdlib.h>
#include "matrixio.h"

void printMatrix(Matrix A) {
    printf("Matrix [%dx%d] = \n", A.rows, A.cols);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            printf("\t%0.3f", A.matrix[i * A.cols + j]);
        }
        printf("\n");
    }
}

void printMatrixToFile(Matrix A, char *filename) {
    FILE *fp = fopen(filename, "w");
    for (int i = 0; i < A.rows; i++) {
        fprintf(fp, "%f", A.matrix[i * A.cols]);
        for (int j = 1; j < A.cols; j++) {
            fprintf(fp, "\t%.15f", A.matrix[i * A.cols + j]);
        }
        fprintf(fp, "\n");
    }
}

Matrix buildArrayFromFile(char *filename) {
    Matrix A;
    FILE *fp = fopen(filename, "r");
    int x = 1, y = 1;
    char c;

    if (fp == NULL) {
        printf("File not found, exiting.\n");
        exit(-2);
    }

    while ((c = (char) fgetc(fp)) != EOF) {
        if (c == '\t') y++;
        else if (c == '\n') break;
    }
    while ((c = (char) fgetc(fp)) != EOF) {
        if (c == '\n')
            x++;
    }

    A.rows = x;
    A.cols = y;
    A.matrix = malloc(y * x * sizeof(double));

    rewind(fp);

    char buffer[30];
    int i = 0, buf_i = 0;
    while ((c = (char) fgetc(fp)) != EOF) {
        if (c == '\t' || c == '\n') {
            buffer[buf_i] = '\0';
            A.matrix[i] = atof(buffer);
            buf_i = 0;
            i++;
        } else {
            buffer[buf_i++] = c;
        }
    }

    fclose(fp);

    return A;
}
