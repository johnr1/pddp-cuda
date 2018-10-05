#ifndef PDDP_MATRIX_H
#define PDDP_MATRIX_H

struct Matrix{
    int rows, cols;
    double* matrix;
};

typedef struct Matrix Matrix;

Matrix zeros(int , int);
Matrix ones(int , int);
Matrix createAndInitializeMatrix(int, int, double);
Matrix calculateAverageVector(Matrix);
double norm(Matrix);

#endif //PDDP_MATRIX_H
