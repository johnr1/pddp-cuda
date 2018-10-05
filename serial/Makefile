all: build/main.o build/file_io.o build/cpddp.o build/matrix.o
	gcc build/*.o -o bin/pddp -lm -O3 -std=c99

build/main.o: src/main.c 
	gcc -o build/main.o src/main.c -c -O3 -std=c99

build/file_io.o: src/file_io.c include/file_io.h
	gcc -o build/file_io.o src/file_io.c -c -O3 -std=c99

build/cpddp.o: src/cpddp.c include/cpddp.h
	gcc -o build/cpddp.o src/cpddp.c -c -O3 -std=c99

build/matrix.o: src/matrix.c include/matrix.h
	gcc -o build/matrix.o src/matrix.c -c -O3 -std=c99

clean:
	rm build/*.o bin/pddp
