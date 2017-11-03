#!/usr/bin/python

import argparse, random

def main():
    args = parse_arguments()
    dim_x = args.x
    dim_y = args.y
    file_name = args.file
    seed = args.s

    random.seed(seed)
    generate_matrix(dim_x, dim_y, file_name)


def generate_matrix(dim_x, dim_y, file_name):
    fhandle = open(file_name, "w")

    for i in range(0, dim_x):
        fhandle.write(str(random.random()))
        for j in range(0, dim_y -1):
            fhandle.write('\t' + str(random.random()))
        fhandle.write('\n')

    fhandle.close()



def parse_arguments():
    parser = argparse.ArgumentParser(description='Generates matrix for the pddp algorithm ' +
                                     'as a text file. Columns are seperated by tabs (\\t) and '+
                                     'rows by newlines (\\n). ' +
                                     'Same seed and size always produces same matrix.')

    parser.add_argument('--cols', '-x', action="store", dest="x", type=int,
                        metavar='N', help="number of rows", default="3")

    parser.add_argument('--rows', '-y', action="store", dest="y", type=int,
                        metavar='N', help="number of columns", default="100")

    parser.add_argument('--seed', '-s', action="store", dest="s", type=int,
                        metavar='N', help="seed for the random number generator", default="1")

    parser.add_argument('file', action="store", metavar='file', help="File name to save ")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
