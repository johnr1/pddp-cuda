#ifndef PDDP_MATRIX_H
#define PDDP_MATRIX_H

struct Matrix{
    unsigned long long rows, cols;
    double* matrix;
};

typedef struct Matrix Matrix;

Matrix calculateAverageVector(Matrix);
Matrix calculateMean(Matrix, Matrix);
Matrix calculateAtA(Matrix);
double calculateAtAElement(Matrix, unsigned long long, unsigned long long);
Matrix matrixMultiply(Matrix, Matrix);
Matrix ones(unsigned long long , unsigned long long);
double norm(Matrix);
Matrix divideMatrixByScalar(Matrix, double);
Matrix matrixSubtract(Matrix, Matrix);


#endif //PDDP_MATRIX_H
