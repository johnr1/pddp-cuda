#include "../include/matrix.h"
#include "../include/cpddp.h"
#include "../include/file_io.h"
#include <math.h>

Matrix pddpStep(Matrix M) {
    double e = 10e-6;
    Matrix w = calculateAverageVector(M);
    Matrix x = ones(M.cols, 1);
    Matrix xNext;

    double varianceNorm;
    do {
        xNext = calculateNextx(M, w, x);
        varianceNorm = calculateVarianceNorm(xNext, x);
        free(x.matrix);
        x = xNext;
    } while (varianceNorm > e);
    return x;
}

Matrix calculateNextx(Matrix M, Matrix w, Matrix x) {
    Matrix xNext = zeros(M.cols, 1);
    Matrix xt =  zeros(M.rows, 1);

    for(int i=0; i<M.rows; i++) {
        for(int j=0; j < M.cols; j++){
            xt.matrix[i] += (M.matrix[i*M.cols+j] - w.matrix[i])*x.matrix[j];
        }
    }

    for(int i=0; i<M.cols; i++) {
        for(int j=0; j < M.rows; j++){
            xNext.matrix[i] += (M.matrix[j*M.cols+i]  - w.matrix[j]) * xt.matrix[j];
        }
    }

    double norma = norm(xNext);

    for (unsigned long long i = 0; i < xNext.rows; ++i) {
        xNext.matrix[i] /= norma;
    }

    free(xt.matrix);

    return xNext;
}


double calculateVarianceNorm(Matrix A, Matrix B)
{
    double powerSum = 0;
    for(unsigned long long i=0; i< A.cols*A.rows; i++){
        powerSum += pow(A.matrix[i] - B.matrix[i], 2);
    }

    return sqrt(powerSum);
}