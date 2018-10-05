#ifndef PDDP_CPDDP_H
#define PDDP_CPDDP_H

Matrix pddpStep(Matrix);

Matrix calculateNextx(Matrix, Matrix, Matrix);
double calculateVarianceNorm(Matrix, Matrix);

#endif //PDDP_CPDDP_H
