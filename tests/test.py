#!/usr/bin/env python
import argparse
from numpy.testing import assert_almost_equal
from subprocess import check_call as shell_exec

expected_results = []
test_results = []
implementation = ""


def compile(implementation):
    print "[i] Compiling"
    shell_exec("cd ../" + implementation + " && make", shell=True)

def clean(implementation):
    print "[i] Cleaning the mess"
    shell_exec("cd ../" + implementation + "/bin && rm result?.csv", shell=True)
    shell_exec("cd ../" + implementation + "/build && rm *.o", shell=True)

# Datasets too large to add to the git repository
def execute(implementation, test):
    print "[i] Executing test " + str(test)
    shell_exec("cd ../" + implementation + "/bin && ./pddp /home/datasets/dataset" + str(test) + ".csv result" + str(test) + ".csv", shell=True)

def read_files(implementation, test):
    global expected_results
    global test_results
    print "[i] Reading Files"
    with open("results/results" + str(test) + ".csv") as f:
        expected_results = map(float, f)
    with open("../" + implementation + "/bin/result" + str(test) + ".csv") as f:
        test_results = map(float, f)
    
def test(implementation, decimals):
    print "[i] Running tests with " + str(decimals) + " decimals accuracy"
    try:
        assert_almost_equal(expected_results, test_results, decimals)
        print "[+] Test OK!"
    except AssertionError as e:
        try:
            op_expected_results = [ -x for x in expected_results]
            assert_almost_equal(op_expected_results, test_results, decimals)
            print "[+] Test OK!"
        except AssertionError as e2:
            print "\033[1;31m" + str(e2) + "\033[0;0m";    
    


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compiles and runs the specified pddp implementation with given dataset and checks the accuracy')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("-t", "--test", action="store", type=int, choices=[1, 2, 3], help="Dataset test to run")
    group.add_argument("-a", "--all", help="Run all dataset tests", action="store_true")

    parser.add_argument("impl", action="store", choices=["serial", "naive", "shared", "shared-optimized"], help="Implementation of pddp to run the test.")

    parser.add_argument("-d", "--decimals", action="store", type=int, help="Decimal point accuracy", default="5")
    parser.add_argument("--clean", help="Cleans the mess", action="store_true")
    
    args = parser.parse_args()
    return args
    
if __name__=="__main__":
    args = parse_arguments()
    compile(args.impl)

    if args.all:
        for i in xrange(1,4):
            execute(args.impl, i)
            read_files(args.impl, i)
            test(args.impl, args.decimals)
    else:
        execute(args.impl, args.test)
        read_files(args.impl, args.test)
        test(args.impl, args.decimals)

    if args.clean:
        clean(args.impl)
