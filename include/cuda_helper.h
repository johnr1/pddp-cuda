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


#define BLOCK_SIZE 32
#define S_BLOCK_SIZE 512


#define GRID_Y 2   //IMPORTANT THERE IS A BUG here
#define GRID_X 512 //Kai genika otan allazoume times


#endif //CU_HELPER_H
