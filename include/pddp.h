#ifndef PDDP_H
#define PDDP_H

void calculateCxNorm(Matrix, Matrix);

__global__ void calculateAverageVector(Matrix, Matrix);
__global__ void initialize(Matrix, double);
__global__ void subtractAndMultiply(Matrix M, Matrix w, Matrix x, Matrix r);
__global__ void subtractAndMultiplyTranspose(Matrix M, Matrix w, Matrix x, Matrix r);


__global__ void matrixMultiplication(Matrix, Matrix, Matrix);
__global__ void divMatrixWithNorm(Matrix, Matrix);
__global__ void subtractMatrix(Matrix, Matrix);


#endif //PDDP_H
