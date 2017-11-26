#include <cuda.h>
#include <stdio.h>
#include "../include/cuda_helper.h"
#include "../include/matrix.h"
#include "../include/pddp.h"


__global__ void initialize(Matrix m, double value){
    int i = blockIdx.x * blockDim.x + threadIdx.x;    
    if(i < m.rows*m.cols){
        m.matrix[i] = value;
    }
}


// TODO Can be optimized
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



/* 
 * *****************
 * (A - w) * x CALCULATION
 * *****************
 */

// TODO Code needs major refactoring
__global__ void mul_reduce(Matrix M, Matrix w, Matrix x, Matrix temp) {
    __shared__ double sh[GRID_Y][GRID_X];
    unsigned int tid = threadIdx.x;
    unsigned int yid = threadIdx.y;
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

    int size = blockIdx.x == (gridDim.x-1) ? (M.cols % blockDim.x) : blockDim.x;

    if(col >= M.cols || col >= x.rows || row >= M.rows || tid >= size)
        return;

    sh[yid][tid] = (M.matrix[row * M.cols + col] - w.matrix[row]) * x.matrix[col]; 
    __syncthreads();

    for(unsigned int s=1; s < size; s *= 2) {
        if (tid % (2*s) == 0 && tid+s < size) {
            sh[yid][tid] += sh[yid][tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 ){
        temp.matrix[row * temp.cols + blockIdx.x] = sh[yid][tid];
    } 
}

__global__ void reduce(Matrix M, Matrix temp) {
    __shared__ double sh[GRID_Y][GRID_X];
    unsigned int tid = threadIdx.x;
    unsigned int yid = threadIdx.y;
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;


    int size = blockIdx.x == gridDim.x-1 ? M.cols % blockDim.x : blockDim.x;

    if(col >= M.cols || row >= M.rows || tid >= size)
        return;

    sh[yid][tid] = M.matrix[row * M.cols + col];
    __syncthreads();

    for(unsigned int s=1; s < size; s *= 2) {
        if (tid % (2*s) == 0 && tid+s < size) {
            sh[yid][tid] += sh[yid][tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 ){
        temp.matrix[row * temp.cols + blockIdx.x] = sh[yid][tid];
    } 
}


void subtractAndMultiply(Matrix M, Matrix w, Matrix x, Matrix mulTemp, Matrix mulTemp2, Matrix temp) {
    mulTemp.cols = M.cols/GRID_X + 1;
    mulTemp2.cols = M.cols/GRID_X + 1;

    dim3 dimGrid(M.cols/GRID_X+1, M.rows/GRID_Y+1, 1);
    dim3 dimBlock(GRID_X, GRID_Y, 1);
    mul_reduce<<<dimGrid, dimBlock>>>(M, w, x, mulTemp); 

    if(dimGrid.x == 1){
        int grid = temp.cols * temp.rows  / S_BLOCK_SIZE + 1;
        copyMatrix<<<grid,S_BLOCK_SIZE>>>(mulTemp, temp);
        
        cudaCheckError();
        return;
    }
    
    do{
        dimGrid.x = dimGrid.x/GRID_X + 1;
        mulTemp2.cols = mulTemp.cols/GRID_X + 1;
        reduce<<<dimGrid, dimBlock>>>(mulTemp, mulTemp2); 

        Matrix juggler = mulTemp;
        mulTemp = mulTemp2;
        mulTemp2 = juggler;
        
    } while(dimGrid.x > 1);
    
    cudaCheckError();
    int grid = temp.cols * temp.rows  / S_BLOCK_SIZE + 1;
    copyMatrix<<<grid,S_BLOCK_SIZE>>>(mulTemp, temp);
    cudaCheckError();

}

__global__ void copyMatrix(Matrix A, Matrix B){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<A.cols* A.rows && i<B.cols*B.rows)
        B.matrix[i] = A.matrix[i];

}


// TODO Can be optimized
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





/* 
 * *****************
 * MATRIX SUBTRACT - MATRIX DIVIDE
 * *****************
 */


__global__ void subtractMatrix(Matrix d_xNext, Matrix d_x){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i > d_x.rows)
        return;

    d_x.matrix[i] = d_xNext.matrix[i] - d_x.matrix[i];

}


__global__ void divMatrixWithNorm(Matrix x, Matrix xNext) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i < xNext.rows)
        xNext.matrix[i] = xNext.matrix[i] / x.matrix[0];
}




/* 
 * *****************
 * NORM CALCULATION
 * *****************
 */

__global__ void dev_norm_calc(Matrix in, Matrix out, int limit, double* varianceNorm) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    int size = blockIdx.x == gridDim.x-1 ? limit % S_BLOCK_SIZE : blockDim.x; //O ipologismos tou size itan lathos

    if(i >= limit || tid >= size)
        return;

    out.matrix[i] = in.matrix[i] * in.matrix[i]; 
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=1; s < size; s *= 2) {
        if (tid % (2*s) == 0 && tid+s < size) {
            out.matrix[i] += out.matrix[i + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0){
        out.matrix[blockIdx.x] = sqrt(out.matrix[blockIdx.x * blockDim.x]); //edw itan episis lathos
        *varianceNorm = out.matrix[blockIdx.x];
    } 
}


void norm(Matrix x, Matrix *temp, Matrix *temp2, double* varianceNorm) {
    int threads = S_BLOCK_SIZE;
    int blockSize = x.rows / S_BLOCK_SIZE + 1;
    dev_norm_calc<<<blockSize, threads>>>(x, *temp, x.rows, varianceNorm); 
    if(blockSize == 1){
        return;
    }

    do{
        int prevBlock = blockSize;
        blockSize = blockSize/threads + 1;
        dev_norm_calc<<<blockSize, threads>>>(*temp, *temp2, prevBlock, varianceNorm); 
        
        double *juggler = (*temp).matrix;
        (*temp).matrix = (*temp2).matrix;
        (*temp2).matrix = juggler;
    } while(blockSize > 1);
    
    cudaCheckError();
}


/* 
 * *****************
 * UNUSED FUNCTIONS (AtomicAdd, MatMul)
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

    //dim3 dimGrid((d_C.cols-1)/BLOCK_SIZE+1, (d_C.rows-1)/BLOCK_SIZE+1, 1);
    //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    //matrixMultiplication<<<dimGrid, dimBlock>>>(d_M, d_M, d_C);
__global__ void matrixMultiplication(Matrix d_A, Matrix d_B, Matrix d_C) {
    __shared__ double sharedMemA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double sharedMemB[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    double pValue = 0;

    for (int i=0; i < (d_A.cols-1)/BLOCK_SIZE+1; i++) {
       if (row < d_A.rows && i*BLOCK_SIZE+tx < d_A.cols)
          sharedMemA[ty][tx] = d_A.matrix[row*d_A.cols + i*BLOCK_SIZE+tx];
       else
          sharedMemA[ty][tx] = 0;
       if (col < d_B.cols && i*BLOCK_SIZE+ty < d_B.rows)
          sharedMemB[ty][tx] = d_B.matrix[(i*BLOCK_SIZE+ty)*d_B.cols+col];
       else
          sharedMemB[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < BLOCK_SIZE; ++k)
          pValue += sharedMemA[ty][k] * sharedMemB[k][tx];
       __syncthreads();
    }
    if (row < d_C.rows && col < d_C.cols)
        d_C.matrix[row*d_C.cols+col] = pValue;

}


