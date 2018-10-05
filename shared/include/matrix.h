#ifndef PDDP_MATRIX_H
#define PDDP_MATRIX_H

struct Matrix{
    int rows, cols;
    double* matrix;
};
typedef struct Matrix Matrix;

// Allocation functions
Matrix matrixDeviceMalloc(int, int);
Matrix matrixHostMalloc(int, int);


#endif //PDDP_MATRIX_H
