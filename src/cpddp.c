#include "../include/matrix.h"
#include "../include/cpddp.h"
#include <stdlib.h>


Matrix pddpStep(Matrix M) {
    double e = 2.2204e-16;
    Matrix w = calculateAverageVector(M);
    Matrix A = calculateMean(M, w);
    Matrix C = calculateAtA(A);
    Matrix x = ones(C.rows,1);
    Matrix xNext;
    double varianceNorm;
    do {
        Matrix Cx = matrixMultiply(C, x);
        double norma = norm(Cx);
        xNext = divideMatrixByScalar(Cx, norma);
        free(Cx.matrix);

        Matrix variance = matrixSubtract(xNext, x);
        varianceNorm = norm(variance);
        free(variance.matrix);
        free(x.matrix);
        x = xNext;
        //printf("%0.20f\n", varianceNorm);

    } while(varianceNorm > e);
    return x;
}
