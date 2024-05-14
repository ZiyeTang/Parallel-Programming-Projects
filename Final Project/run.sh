#!/bin/bash

clean() {
    echo "Cleaning up..."
    ./cleanfile.sh
    rm -rf ./m1 ./m2 ./m3 ./final *.out *.err outfile
}

build() {
    echo "Building the project..."
    ./cleanfile.sh
    cmake ./project/ && make -j8
    ./cleanfile.sh
}


case "$1" in
    clean) clean ;;
    build) build ;;
    *) echo "Usage: $0 {clean|build}" ;;
esac
