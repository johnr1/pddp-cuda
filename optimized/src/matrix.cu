#include "../include/matrix.h"
#include <stdlib.h>
#include <cuda.h>
#include "../include/cuda_helper.h"

Matrix matrixDeviceMalloc(int rows, int cols){
    Matrix m;
    m.cols = cols;
    m.rows = rows;
    cudaMalloc((void**)&m.matrix, m.cols*m.rows*sizeof(double));
    cudaCheckError();

    return m;
}

Matrix matrixHostMalloc(int rows, int cols){
    Matrix m;
    m.cols = cols;
    m.rows = rows;
    m.matrix = (double*) malloc(m.cols*m.rows*sizeof(double));

    return m;
}

