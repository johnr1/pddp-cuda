# Principal Direction Divisive Partitioning (PDDP) 

## Description 

PDDP is clustering algorithm, which takes as input a set of vectors in n x m matrix format, usually m >> n, where each row of it, is a vector. Using the input matrix, it calculates the 'principal component', an eigenvector, as output. 

The first step of the pddp algorithm is to calculate the average vector of all elements of matrix M as follows:

> w = (1/m) * ( d<sub>1</sub> + d<sub>2</sub> + ... + d<sub>m</sub> )

The next step is to calculate the matrix C:

> C = (M − w * e<sup>T</sup>)<sup>T</sup> * (M − w * e<sup>T</sup>) = A<sup>T</sup> * A,  
> where A = (M − w * e<sup>T</sup>)
> and e = [111 ... 1]<sup>T</sup>
 
In a nutshell, from every column of the input matrix M, the average of the input (matrix A) should be subtracted and the result will be multiplied with the transpose of A. 

After that, the principal component (the eigenvector) approximation is calculated iteratively. Starting with a vector x<sub>0</sub> = [111...1], the final result is approached as follows:

> x<sub>k+1</sub> = (C * x<sub>k</sub>) / || C * x<sub>k</sub> ||

The algorithm converges, when a new x<sub>k+1</sub> vector has a small difference from the previous x<sub>k</sub>. This eigenvector is used to separate the input data in two subsets A and B as follows. The eigenvector is has positive and negative values. Each value of it corresponds to a vector in the input matrix. The first value corresponds to the first vector the second to the second etc. Subsets A and B are comprised of the vectors matched with a positive and negative values respectively.

Further clustering can be achieved by giving one of these subsets as input to the algorithm.

More information can be found [here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4267&rep=rep1&type=pdf)

## Implementations

Three implementations of the pddp algorithm are provided in total.

| Version                               | Directory         | Description                                         		    |
| ------------------------------------- | ----------------- |-------------------------------------------------------------- |
| Serial                              	| serial/           | A serial implementation in C.                   	  		    |
| Naive implementation in CUDA          | naive/            | Naive implementation in CUDA using exclusively global memory. |
| Optimized implementation in CUDA 		| optimized/        | Optimal implementation in CUDA using shared memory.           |


## Compile Instructions
Change directory to the desired implementation, and run:
> make 

The executable will be placed in the 'bin/' directory and the object files in the 'build/'.


## Run
All the implementations produce an executable called 'pddp' in the 'bin/' directory. It is executed as follows:

> ./pddp input_file [output_file]

* 'input_file' a file in .csv format, each column represents a vector
* 'output_file' it's the final eigenvector in .csv format, default filename is 'output.csv'


## Tests
Test datasets are not uploaded at the moment due to their large size. The test.py script can be used by providing your own datasets and results.

### Examples
* Execute and test the 'serial' implementation with the 1st dataset
> ./test.py -t1 serial

* Execute and test the 'shared' implementation with all the datasets and clean afterwards (object files and results).
> ./test.py --all --clean shared

* Execute and test the 'shared' implementation with the 3rd dataset up to the 6th decimal place.
> ./test.py -d6 -t3 shared

