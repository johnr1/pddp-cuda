#include <cuda.h>
#include <stdio.h>
#include "../include/cuda_helper.h"
#include "../include/matrix.h"
#include "../include/pddp.h"


/**
 * ----------------
 * w <- mean(M')'
 * -----------------
 */
__global__ void calculateAverageVector(Matrix d_M, Matrix d_w){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < d_M.rows) {
        double sum = 0;
        for (int j = 0; j < d_M.cols; j++) {
            sum += d_M.matrix[i * d_M.cols + j];
        }
        d_w.matrix[i] = sum / d_M.cols;
    }
}


/**
 * ----------------
 * temp <- (M - w) * x
 * -----------------
 */
__global__ void subtractAndMultiply(Matrix M, Matrix w, Matrix x, Matrix temp) {
    __shared__ double sh[GRID_Y][GRID_X];
    unsigned int tid = threadIdx.x;
    unsigned int yid = threadIdx.y;
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row >= M.rows)
        return;

    if(col < M.cols)
        sh[yid][tid] = (M.matrix[row * M.cols + col] - w.matrix[row]) * x.matrix[col]; 
    else
        sh[yid][tid] = 0;
   
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sh[yid][tid] += sh[yid][tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 ){
        atomicAdd(&temp.matrix[row], sh[yid][tid]);
    } 
}


/**
 * ----------------
 * r <- (M - w)' * x 
 * -----------------
 */
 __global__ void subtractAndMultiplyTranspose(Matrix M, Matrix w, Matrix x, Matrix r) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0;
    if(row < r.rows) {
        for(int col=0; col< M.rows; col++){
            sum += (M.matrix[row + M.cols*col] - w.matrix[col]) * x.matrix[col];
        }        
        r.matrix[row] = sum;
    }
}


/**
 * -----------------
 * NORM CALCULATION
 * -----------------
 */
void calculateNorm(Matrix x, Matrix *temp, Matrix *temp2, double* varianceNorm) {
    int blockSize = x.rows / S_BLOCK_SIZE + 1;
    int threads = S_BLOCK_SIZE;

    normCalculationKernel<<<blockSize, threads>>>(x, *temp, x.rows, varianceNorm); 

    while(blockSize > 1) {
        int prevBlock = blockSize;
        blockSize = blockSize/threads + 1;
        normCalculationKernel<<<blockSize, threads>>>(*temp, *temp2, prevBlock, varianceNorm); 
        
        double *juggler = (*temp).matrix;
        (*temp).matrix = (*temp2).matrix;
        (*temp2).matrix = juggler;
    }
    
    cudaCheckError();
}


__global__ void normCalculationKernel(Matrix in, Matrix out, int limit, double* varianceNorm) {
    __shared__ double sh[S_BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i < limit)
        sh[tid] = in.matrix[i] * in.matrix[i]; 
    else
        sh[tid] = 0;
        
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sh[tid] += sh[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0){
        double result = sqrt(sh[tid]);
        out.matrix[blockIdx.x] = result; 
        *varianceNorm = result;
    } 
}


/**
 * ------------------
 * MATRIX SUBTRACT - MATRIX DIVIDE - MATRIX INIT
 * ------------------
 */
__global__ void subtractMatrix(Matrix d_xNext, Matrix d_x){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < d_x.rows)
        d_x.matrix[i] = d_xNext.matrix[i] - d_x.matrix[i];

}


__global__ void divMatrixWithNorm(Matrix x, Matrix xNext) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i < xNext.rows)
        xNext.matrix[i] = xNext.matrix[i] / x.matrix[0];
}

__global__ void initialize(Matrix m, double value){
    int i = blockIdx.x * blockDim.x + threadIdx.x;    
    if(i < m.rows*m.cols){
        m.matrix[i] = value;
    }
}


/* 
 * *****************
 * DOUBLE ATOMIC ADD
 * *****************
 */
 __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}



