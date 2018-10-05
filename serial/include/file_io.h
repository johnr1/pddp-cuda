#ifndef FILE_IO
#define FILE_IO

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include "../include/matrix.h"

#define MAX_CHAR_PER_LINE 128

#define NONE 0
#define FIRST 1
#define LAST 2
#define BOTH 3


double* file_read(char *filename,int  *numObjs,int  *numCoords);



void print(Matrix);
void print_to_file(Matrix, char*);


#endif

