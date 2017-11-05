cmake_minimum_required(VERSION 3.8)
project(pddp)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS -lm)
set(H)

set(SOURCE_FILES src/main.c src/matrixio.c src/cpddp.c include/matrix.h include/matrixio.h include/cpddp.h src/matrix.c)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ../bin)

add_executable(exe ${SOURCE_FILES})