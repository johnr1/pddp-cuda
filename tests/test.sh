#!/usr/bin/env bash


#./exe large.txt
if [[ $# == 0 ]]; then
    echo "Usage: $1 [filename|--all]"
    echo "  --all: run all tests"
    exit 1
fi

function test {
    start_time=`date +%s`
    ../bin/exe "/home/datasets/dataset$1.csv" "result.mat"  
    end_time=`date +%s`
    diff result.mat "./results/o_result$1.csv"  
    if [[ $? -eq 0 ]]; then
        echo "[+] Test $1 OK! [`expr $end_time - $start_time`s]"
    else
        echo "[-] Test $1 FAILED! [`expr $end_time - $start_time`s]"
    fi
}

# Compile
cd ../bin
make
cd ../tests
if [[ $1 == "--all" ]]; then
    for i in `seq 1 3`;
    do
        test $i;
    done
else
    test $1;
fi

rm result.mat
