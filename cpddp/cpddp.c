#include "../matrix.h"
#include "cpddp.h"
#include "../io/matrixio.h"
#include <stdlib.h>
#include <stdio.h>


Matrix pddpStep(Matrix M) {
    double e = 0.0000000000000000000001;
    Matrix w = calculateAverageVector(M);
    Matrix A = calculateMean(M, w);
    printMatrix(A);
    Matrix C = calculateAtA(A); //PROBLEM
    printMatrix(C);
    exit(-1);


    Matrix x = ones(C.rows,1);
    Matrix xNext;
    double xfactorNorm;
    do {
        Matrix Cx = matrixMultiply(C, x);
        double norma = norm(Cx);
        printMatrix(C);
        xNext = divideMatrixByScalar(Cx, norma);
        free(Cx.matrix);

        Matrix xfactor = matrixSubtract(xNext, x);
        xfactorNorm = norm(xfactor);
        free(xfactor.matrix);
        free(x.matrix);
        x = xNext;
        printf("%0.29f\n", xfactorNorm);

    } while(xfactorNorm>e);
    return x;
}
