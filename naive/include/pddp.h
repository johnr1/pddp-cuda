#ifndef PDDP_H
#define PDDP_H

__global__ void calculateAverageVector(Matrix, Matrix);

__global__ void subtractAndMultiply(Matrix M, Matrix w, Matrix x, Matrix temp);
__global__ void subtractAndMultiplyTranspose(Matrix M, Matrix w, Matrix x, Matrix r);

__global__ void initialize(Matrix, double);
__global__ void divMatrixWithNorm(Matrix, Matrix);
__global__ void subtractMatrix(Matrix, Matrix);

void calculateNorm(Matrix, Matrix*, Matrix*, double*);
__global__ void normCalculationKernel(Matrix, Matrix, int, double*);


#endif //PDDP_H
