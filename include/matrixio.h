#ifndef PDDP_MATRIXIO_H
#define PDDP_MATRIXIO_H
#include "matrix.h"

Matrix matrixFileValidator(char*);
Matrix buildMatrixFromFile(char *);
void printMatrix(Matrix);
void printMatrixToFile(Matrix, char*);

#endif //PDDP_MATRIXIO_H
