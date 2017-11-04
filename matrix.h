#ifndef PDDP_MATRIX_H
#define PDDP_MATRIX_H

struct Matrix{
    int rows, cols;
    double* matrix;
};

typedef struct Matrix Matrix;

Matrix calculateAverageVector(Matrix);
Matrix calculateMean(Matrix, Matrix);
Matrix calculateAtA(Matrix);
double calculateAtAElement(Matrix, int, int);
Matrix matrixMultiply(Matrix, Matrix);
Matrix ones(int , int);
double norm(Matrix);
Matrix divideMatrixByScalar(Matrix, double);
Matrix matrixSubtract(Matrix, Matrix);


#endif //PDDP_MATRIX_H
