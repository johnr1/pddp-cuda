cmake_minimum_required(VERSION 3.8)
project(pddp)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS -lm)
set(H)

set(SOURCE_FILES main.c io/matrixio.c cpddp/cpddp.c matrix.h io/matrixio.h cpddp/cpddp.h matrix.c)

add_executable(exe ${SOURCE_FILES})