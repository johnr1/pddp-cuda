#include <stdlib.h>
#include "matrix.h"
#include <math.h>

Matrix ones(int N, int M) {
    Matrix A;
    A.rows = N;
    A.cols = M;
    A.matrix = malloc(M * N * sizeof(double));

    for (int i = 0; i < N * M; ++i) {
        A.matrix[i] = 1;
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

Matrix calculateMean(Matrix M, Matrix w) {
    Matrix A;
    A.cols = M.cols;
    A.rows = M.rows;

    A.matrix = malloc(M.cols * M.rows * sizeof(double));


    for (int i = 0; i < M.cols * M.rows; i++) {
        A.matrix[i] = M.matrix[i] - w.matrix[i / A.cols];
    }

    return A;
}

Matrix calculateAtA(Matrix A) {
    Matrix C;
    C.cols = A.cols;
    C.rows = A.cols;

    C.matrix = malloc(A.cols * A.cols * sizeof(double));

    for (int i = 0; i < A.cols * A.cols; i++) {
        C.matrix[i] = calculateAtAElement(A, i / A.cols, i % A.cols);
    }

    return C;
}

double calculateAtAElement(Matrix A, int x, int y) {
    double sum = 0;
    for (int i = 0; i < A.rows; i++) {
        sum += A.matrix[i * A.cols + x] * A.matrix[i * A.cols + y];
    }

    return sum;
}

Matrix matrixMultiply(Matrix A, Matrix B) {
    Matrix C;
    C.rows = A.rows;
    C.cols = B.cols;

    C.matrix = malloc(C.rows * C.cols * sizeof(double));

    for (int i = 0; i < C.rows; i++) {
        for (int j = 0; j < C.cols; j++) {
            double sum = 0;
            for (int k = 0; k < A.cols; k++) {
                sum += A.matrix[i * A.cols + k] * B.matrix[k * B.cols + j];
            }
            C.matrix[i * C.cols + j] = sum;
        }
    }

    return C;
}

double norm(Matrix x) {
    double sum = 0;
    for (int i = 0; i < x.rows; ++i) {
        sum += pow(x.matrix[i], 2);
    }
    return sqrt(sum);
}

Matrix divideMatrixByScalar(Matrix A, double scalar) {
    Matrix R;
    R.rows = A.rows;
    R.cols = A.cols;

    R.matrix = malloc(R.cols*R.rows*sizeof(double));

    for(int i=0; i< A.cols*A.rows; i++){
        R.matrix[i] = A.matrix[i] / scalar;
    }

    return R;
}

Matrix matrixSubtract(Matrix A, Matrix B) {
    Matrix R;
    R.rows = A.rows;
    R.cols = A.cols;

    R.matrix = malloc(R.cols*R.rows*sizeof(double));

    for(int i=0; i< A.cols*A.rows; i++){
        R.matrix[i] = A.matrix[i] - B.matrix[i];
    }

    return R;
}

