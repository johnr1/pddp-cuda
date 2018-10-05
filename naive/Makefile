all: build/main.o build/file_io.o build/pddp.o build/matrix.o
	nvcc build/*.o --generate-code arch=compute_20,code=sm_20 -o bin/pddp -O3 --use_fast_math	

build/main.o: src/main.cu
	nvcc -o build/main.o src/main.cu --generate-code arch=compute_20,code=sm_20 -dc  -O3 --use_fast_math	

build/file_io.o: src/file_io.cu include/file_io.h
	nvcc -o build/file_io.o src/file_io.cu --generate-code arch=compute_20,code=sm_20 -dc  -O3 --use_fast_math	

build/pddp.o: src/pddp.cu include/pddp.h
	nvcc -o build/pddp.o src/pddp.cu --generate-code arch=compute_20,code=sm_20 -dc  -O3 --use_fast_math	

build/matrix.o: src/matrix.cu include/matrix.h
	nvcc -o build/matrix.o src/matrix.cu --generate-code arch=compute_20,code=sm_20 -dc -O3 --use_fast_math	

clean:
	rm build/*.o bin/pddp
