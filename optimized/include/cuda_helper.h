#ifndef CU_HELPER_H
#define CU_HELPER_H

#include <stdio.h>
#define cudaCheckError() {                               \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}


#define S_BLOCK_SIZE 512

#define GRID_Y 1
#define GRID_X 512  //GRID_X<256 perhaps hits memory limits


#endif //CU_HELPER_H
