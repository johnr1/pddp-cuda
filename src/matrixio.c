#include <stdio.h>
#include <stdlib.h>
#include "../include/matrixio.h"

void printMatrix(Matrix A) {
    printf("Matrix [%llux%llu] = \n", A.rows, A.cols);
    for (unsigned long long i = 0; i < A.rows; i++) {
        for (unsigned long long j = 0; j < A.cols; j++) {
            printf("\t%f", A.matrix[i * A.cols + j]);
        }
        printf("\n");
    }
}

void printMatrixToFile(Matrix A, char *filename) {
    FILE *fp = fopen(filename, "w");
    for (unsigned long long i = 0; i < A.rows; i++) {
        fprintf(fp, "%f", A.matrix[i * A.cols]);
        for (unsigned long long j = 1; j < A.cols; j++) {
            fprintf(fp, "\t%.15f", A.matrix[i * A.cols + j]);
        }
        fprintf(fp, "\n");
    }
}

Matrix matrixFileValidator(char *filename) {
    Matrix A;

    FILE *fp = fopen(filename, "r");
    unsigned long long x = 1, y = 1, totalColumns;
    char c;

    if (fp == NULL) {
        fprintf(stderr,"File not found, exiting.\n");
        exit(1);
    }

    while ((c = (char) fgetc(fp)) != EOF) {
        if (c == '\t' || c == ' ') {
            y++;
            consumeWhitespace(fp);
        } else if (c == '\n') break;
    }

    totalColumns = y;

    y = 1;
    while ((c = (char) fgetc(fp)) != EOF) {
        if (c == '\t' || c == ' ') {
            y++;
            consumeWhitespace(fp);
        }

        if (c == '\n') {
            if (totalColumns != y) {
                fprintf(stderr,"Malformed file.\n");
                exit(2);
            }
            x++;
            y = 1;
        }
    }


    A.cols = totalColumns;
    A.rows = x;
    A.matrix = malloc(x * totalColumns * sizeof(double));

    fclose(fp);

    return A;
}


Matrix buildMatrixFromFile(char *filename) {
    FILE *fp = fopen(filename, "r");

    Matrix A = matrixFileValidator(filename);

    char c, buffer[30];
    unsigned long long i = 0;
    unsigned int buf_i = 0;

    while ((c = (char) fgetc(fp)) != EOF) {
        if (c == '\t' || c == '\n' || c == ' ') {
            buffer[buf_i] = '\0';
            A.matrix[i] = atof(buffer);
            consumeWhitespace(fp);
            buf_i = 0;
            i++;
        } else {
            buffer[buf_i++] = c;
        }
    }

    fclose(fp);

    return A;
}

void consumeWhitespace(FILE *fp){
    char c;
    while ((c = (char) fgetc(fp)) != EOF) {
        if (c != '\t' && c != ' ') {
            ungetc(c, fp);
            break;
        }
    }

}
