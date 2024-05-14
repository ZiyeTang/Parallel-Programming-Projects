#!/bin/bash

# Check if ./CMakeFiles directory exists, then remove it
if [ -d "./CMakeFiles" ]; then
    echo "Removing ./CMakeFiles directory"
    rm -rf "./CMakeFiles"
fi

# Check if ./src directory exists, then remove it
if [ -d "./src" ]; then
    echo "Removing ./src directory"
    rm -rf "./src"
fi

# Check if CMakeCache.txt file exists, then remove it
if [ -e "CMakeCache.txt" ]; then
    echo "Removing CMakeCache.txt file"
    rm -f "CMakeCache.txt"
fi

# Check if Makefile file exists, then remove it
if [ -e "Makefile" ]; then
    echo "Removing Makefile file"
    rm -f "Makefile"
fi

# Check if libece408net file exists, then remove it
if [ -e "libece408net.a" ]; then
    echo "Removing libece408net file"
    rm -f "libece408net.a"
fi

# Check if cmake_install.cmake file exists, then remove it
if [ -e "cmake_install.cmake" ]; then
    echo "Removing cmake_install.cmaket file"
    rm -f "cmake_install.cmake"
fi