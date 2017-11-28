#!/usr/bin/env python
import argparse
from numpy.testing import assert_almost_equal
from subprocess import check_call as shell_exec

expected_results = []
test_results = []

def compile():
    print "[i] Compiling"
    shell_exec("cd ../bin && make", shell=True)

def clean():
    print "[i] Cleaning the mess"
    shell_exec("cd ../bin && rm result?.csv", shell=True)

def execute(test):
    print "[i] Executing test " + str(test)
    shell_exec("cd ../bin && nvprof ./exe /home/datasets/dataset" + str(test) + ".csv result" + str(test) + ".csv", shell=True)

def read_files(test):
    global expected_results
    global test_results
    print "[i] Reading Files"
    with open("results/o_result" + str(test) + ".csv") as f:
        expected_results = map(float, f)
    with open("../bin/result" + str(test) + ".csv") as f:
        test_results = map(float, f)
    
def test(decimals):
    print "[i] Running tests with " + str(decimals) + " decimals accuracy"
    try:
        assert_almost_equal(expected_results, test_results, decimals)
        print "[+] Test OK!"
    except AssertionError as e:
        print "\033[1;31m" + str(e) + "\033[0;0m";    

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compiles and run exe with given dataset and checks the accuracy')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--test", action="store", type=int, choices=[1, 2, 3], help="test to run")
    group.add_argument("-a", "--all", help="Run all tests", action="store_true")

    parser.add_argument("-d", "--decimals", action="store", type=int, help="decimal point accuracy", default="5")
    parser.add_argument("--clean", help="Cleans the mess", action="store_true")
    
    args = parser.parse_args()
    return args
    
if __name__=="__main__":
    args = parse_arguments()
    compile()

    if args.all:
        for i in xrange(1,4):
            execute(i)
            read_files(i)
            test(args.decimals)
    else:
        execute(args.test)
        read_files(args.test)
        test(args.decimals)

    if args.clean:
        clean()
