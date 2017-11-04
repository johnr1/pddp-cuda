#!/usr/bin/env bash



#./exe data/large.txt
if [[ $# == 0 ]]; then
    echo "Usage: $1 [filename|--all]"
    echo "  --all: run all tests"
    exit 1
fi

function test {
    start_time=`date +%s`
    ./exe "data/$1" > /dev/null
    end_time=`date +%s`
    diff result.txt results/$1 > /dev/null
    if [[ $? -eq 0 ]]; then
        echo "[+] Test $1 OK! [`expr $end_time - $start_time`s]"
    else
        echo "[+] Test $1 FAILED! [`expr $end_time - $start_time`s]"
    fi

}

# Compile
cd .. &&
make > /dev/null &&
mv exe tests/exe &&
rm CMakeCache.txt
rm cmake_install.cmake
rm -rf CMakeFiles/
cd tests &&

if [[ $1 == "--all" ]]; then
    tests=$(find data -type f)
    for test in ${tests[@]}
    do
        test $test
    done
else
    test $1;
fi

rm exe