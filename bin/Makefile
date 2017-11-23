all: main.o file_io.o pddp.o matrix.o
	nvcc ./*.o --generate-code arch=compute_20,code=sm_20 -o exe -g 

main.o: ../src/main.cu
	nvcc ../src/main.cu --generate-code arch=compute_20,code=sm_20 -g -dc

file_io.o: ../src/file_io.cu
	nvcc ../src/file_io.cu --generate-code arch=compute_20,code=sm_20 -g -dc

pddp.o:	../src/pddp.cu
	nvcc ../src/pddp.cu --generate-code arch=compute_20,code=sm_20 -g -dc

matrix.o: ../src/matrix.cu
	nvcc ../src/matrix.cu --generate-code arch=compute_20,code=sm_20 -g -dc

clean:
	rm *.o exe
