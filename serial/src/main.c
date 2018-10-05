#include <stdio.h>
#include <stdlib.h>
#include "../include/file_io.h"
#include "../include/cpddp.h"
#include <time.h>

int main(int argc, char *argv[])
{
    double time_elapsed;
    char *input_file, *output_file;
    if (argc < 2){
        printf("Usage: %s input_file [output_file]\n", argv[0]);
        exit(1);
    } else if(argc == 2) {
        input_file = argv[1];
        output_file = "result.csv";
    }
    else{
        input_file = argv[1];
        output_file = argv[2];
    }

    printf("Reading input file...\n");
    Matrix M;
    M.matrix = file_read(input_file, &M.rows, &M.cols);
    printf("Performing calculations...\n");

    clock_t start = clock();

    Matrix eigenVector = pddpStep(M);

    time_elapsed = (double)(clock()-start)/CLOCKS_PER_SEC;
    printf("Calculations finished\n");
    printf("Time elapsed: %lf ms\n", time_elapsed*1000);

    print_to_file(eigenVector, output_file);

    free(M.matrix);
    free(eigenVector.matrix);

    return 0;
}
