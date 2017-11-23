#include <cuda.h>
#include "../include/file_io.h"
#include "../include/matrix.h"
#include "../include/pddp.h"
#include "../include/cuda_helper.h"

__global__ void reduce(Matrix in, Matrix out, int limit, double* varianceNorm) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    int size = limit < blockDim.x ? limit : blockDim.x;

    if(i > in.rows)
        return;

    out.matrix[i] = in.matrix[i] * in.matrix[i]; // wrong ind mallon
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
        out.matrix[blockIdx.x] = sqrt(out.matrix[blockIdx.x]);
        *varianceNorm = out.matrix[blockIdx.x];
        //printf("DEBUG: %f \n", out.matrix[blockIdx.x]);
    } 
}


void norm(Matrix x, Matrix temp, double* varianceNorm) {
    int counter = 1;
    int threads = S_BLOCK_SIZE;
    int blockSize = x.rows / S_BLOCK_SIZE + 1;
    reduce<<<blockSize, threads>>>(x, temp, x.rows, varianceNorm); 
    if(blockSize == 1){
        printf("Recursive reductions: %d.\n", counter);
        return;
    }

    do{
        counter++;
        int prevBlock = blockSize;
        blockSize = blockSize/threads + 1;
        reduce<<<blockSize, threads>>>(temp, temp, prevBlock, varianceNorm); 
    } while(blockSize > 1);

    printf("Recursive reductions: %d.\n", counter);
    
    cudaCheckError();
}



int main(int argc, char* argv[]) {
    const double e = 10e-6;
    if (argc < 2){
        printf("Usage: %s filename\n", argv[0]);
        exit(1);
    }
    printf("Program started\n");
    fflush(stdout);

    // Host
    Matrix M;
    M.matrix = file_read(argv[1], &M.rows, &M.cols);
    printf("File read\n");
    fflush(stdout);
    Matrix x = matrixHostMalloc( M.cols, 1);


    // Device
    Matrix d_M = matrixDeviceMalloc(M.rows, M.cols);
    Matrix d_w = matrixDeviceMalloc(M.rows, 1);
    Matrix d_xNext = matrixDeviceMalloc(M.cols, 1);
    Matrix d_x = matrixDeviceMalloc(M.cols, 1);
    Matrix d_temp = matrixDeviceMalloc(M.cols, 1);
    

    // Transfer M matrix to device
    cudaMemcpy(d_M.matrix, M.matrix, M.cols*M.rows*sizeof(double), cudaMemcpyHostToDevice);
    cudaCheckError();
    

    // Kernels which calculates avg weight vector and initializes d_x
    calculateAverageVector<<<M.rows/S_BLOCK_SIZE + 1, S_BLOCK_SIZE>>>(d_M,d_w); //Populates d_w
    cudaCheckError();

    initialize<<<d_x.rows/S_BLOCK_SIZE + 1, S_BLOCK_SIZE>>>(d_x,1);             //Populated d_x
    cudaCheckError();

    // Allocate Mapped varianceNorm value
    double *varianceNorm, *d_varianceNorm; //1 iteration
    cudaHostAlloc((void **)&varianceNorm, sizeof(double), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **)&d_varianceNorm, varianceNorm, 0);





    /* EXPIRIMENTING 

    Matrix d_temp2 = matrixDeviceMalloc(M.rows, M.cols);
    norm(d_M,d_temp2,d_varianceNorm);
    cudaDeviceSynchronize();
    
    exit(0);

    END */











    printf("Memory allocations finished\n");
    fflush(stdout);

    Matrix tempPointer;
    *varianceNorm = 0;
    do {
        d_temp.rows = M.rows;
        subtractAndMultiply<<<M.rows/S_BLOCK_SIZE + 1, S_BLOCK_SIZE>>>(d_M, d_w, d_x, d_temp);
        subtractAndMultiplyTranspose<<<M.cols/S_BLOCK_SIZE + 1, S_BLOCK_SIZE>>>(d_M, d_w, d_temp, d_xNext);

        norm(d_xNext,d_temp,d_varianceNorm); //d_temp[0] contains norm value
        divMatrixWithNorm<<<(d_xNext.rows/S_BLOCK_SIZE)+1, S_BLOCK_SIZE>>>(d_temp, d_xNext); //Alters d_xNext
        
        d_temp.rows = M.cols;
        subtractMatrix<<<(d_xNext.rows/S_BLOCK_SIZE)+1, S_BLOCK_SIZE>>>(d_xNext, d_x); //Alters d_x
        norm(d_x, d_temp, d_varianceNorm); //makes d_temp[0] the norm value

        tempPointer = d_x; //Jungle pointers
        d_x = d_xNext;
        d_xNext = tempPointer;

        cudaDeviceSynchronize();
    } while(*varianceNorm > e);






    cudaDeviceSynchronize();
    cudaCheckError();
    cudaMemcpy(x.matrix, d_x.matrix, d_x.rows*sizeof(double), cudaMemcpyDeviceToHost);
    cudaCheckError();

    print(x);

    cudaFree(d_M.matrix);
    cudaCheckError();
    cudaFree(d_w.matrix);
    cudaCheckError();
    cudaFree(d_x.matrix);
    cudaCheckError();
    cudaFree(d_xNext.matrix);
    cudaCheckError();
    free(M.matrix);
    free(x.matrix);

    return 0;
}



