cmake_minimum_required(VERSION 3.8)
project(pddp)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES main.c io/matrixio.c)
add_executable(exe ${SOURCE_FILES})