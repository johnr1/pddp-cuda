#ifndef PDDP_H
#define PDDP_H

void calculateCxNorm(Matrix, Matrix);

__global__ void calculateAverageVector(Matrix, Matrix);
__global__ void initialize(Matrix, double);
__global__ void subtractAndMultiplyTranspose(Matrix M, Matrix w, Matrix x, Matrix r);


__global__ void matrixMultiplication(Matrix, Matrix, Matrix);
__global__ void divMatrixWithNorm(Matrix, Matrix);
__global__ void subtractMatrix(Matrix, Matrix);

__device__ double atomicAdd(double*, double);

__global__ void dev_norm_calc(Matrix, Matrix, int, double*);
void norm(Matrix, Matrix*, Matrix*, double*);

void subtractAndMultiply(Matrix, Matrix, Matrix, Matrix, Matrix, Matrix);
__global__ void mul_reduce(Matrix, Matrix, Matrix, Matrix);
__global__ void reduce(Matrix, Matrix);
__global__ void copyMatrix(Matrix, Matrix);





#endif //PDDP_H
