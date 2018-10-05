#include <stdlib.h>
#include "../include/matrix.h"
#include <math.h>

Matrix zeros(int N, int M) {
    return createAndInitializeMatrix(N, M, 0);
}

Matrix ones(int N, int M) {
    return createAndInitializeMatrix(N, M, 1);
}

Matrix createAndInitializeMatrix(int N, int M, double value) {
    Matrix A;
    A.rows = N;
    A.cols = M;
    A.matrix = malloc(M * N * sizeof(double));

    for (int i = 0; i < N * M; ++i) {
        A.matrix[i] = value;
    }

    return A;
}

Matrix calculateAverageVector(Matrix M) {
    Matrix w;
    w.cols = 1;
    w.rows = M.rows;
    w.matrix = malloc(M.rows * sizeof(double));

    for (int i = 0; i < M.rows; i++) {
        double sum = 0;
        for (int j = 0; j < M.cols; j++) {
            sum += M.matrix[i * M.cols + j];
        }
        w.matrix[i] = sum / M.cols;
    }

    return w;
}

double norm(Matrix x) {
    double sum = 0;
    for (int i = 0; i < x.rows; ++i) {
        sum += pow(x.matrix[i], 2);
    }
    return sqrt(sum);
}
