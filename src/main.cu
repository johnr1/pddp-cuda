#include <cuda.h>
#include "../include/file_io.h"
#include "../include/matrix.h"
#include "../include/pddp.h"
#include "../include/cuda_helper.h"


int main(int argc, char* argv[]) {
    const double e = 10e-6;
    char *input_file, *output_file;
    if (argc < 2){
        printf("Usage: %s input_file [output_file]\n", argv[0]);
        exit(1);
    } else if(argc == 2) {
        input_file = argv[1];
        output_file = "result.mat";
    }
    else{
        input_file = argv[1];
        output_file = argv[2];
    }
    
    printf("Program started\n");
    fflush(stdout);

    // Host
    Matrix M;
    M.matrix = file_read(input_file, &M.rows, &M.cols);
    printf("File read\n");
    fflush(stdout);
    Matrix x = matrixHostMalloc( M.cols, 1);


    // Device
    Matrix d_M = matrixDeviceMalloc(M.rows, M.cols);
    Matrix d_w = matrixDeviceMalloc(M.rows, 1);
    Matrix d_xNext = matrixDeviceMalloc(M.cols, 1);
    Matrix d_x = matrixDeviceMalloc(M.cols, 1);
    Matrix d_temp = matrixDeviceMalloc(M.cols, 1);
    Matrix d_temp2 = matrixDeviceMalloc(M.cols,1);
    Matrix d_mulTemp = matrixDeviceMalloc(M.rows, M.cols/GRID_X + 1);
    Matrix d_mulTemp2 = matrixDeviceMalloc(M.rows, M.cols/GRID_X/GRID_X + 1);
    

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


    printf("Memory allocations finished\n");
    fflush(stdout);

    Matrix tempPointer;
    *varianceNorm = 0;
    do {
        d_temp.rows = M.rows;
        subtractAndMultiply(d_M, d_w, d_x, d_mulTemp, d_mulTemp2, d_temp);

        subtractAndMultiplyTranspose<<<M.cols/S_BLOCK_SIZE + 1, S_BLOCK_SIZE>>>(d_M, d_w, d_temp, d_xNext);

        norm(d_xNext,&d_temp,&d_temp2,d_varianceNorm); //d_temp[0] contains norm value
        divMatrixWithNorm<<<(d_xNext.rows/S_BLOCK_SIZE)+1, S_BLOCK_SIZE>>>(d_temp, d_xNext); //Alters d_xNext
        
        d_temp.rows = M.cols;
        subtractMatrix<<<(d_xNext.rows/S_BLOCK_SIZE)+1, S_BLOCK_SIZE>>>(d_xNext, d_x); //Alters d_x
        norm(d_x, &d_temp, &d_temp2, d_varianceNorm); //makes d_temp[0] the norm value

        tempPointer = d_x; //Jungle pointers
        d_x = d_xNext;
        d_xNext = tempPointer;

        cudaDeviceSynchronize();
    } while(*varianceNorm > e);


    cudaDeviceSynchronize();
    cudaCheckError();
    cudaMemcpy(x.matrix, d_x.matrix, d_x.rows*sizeof(double), cudaMemcpyDeviceToHost);
    cudaCheckError();

    print_to_file(x, output_file);//printing to file in order to both check values and print debug info

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



