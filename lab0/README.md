# Lab 0

## Objective

The purpose of this lab is to help you become familiar with using the development system for this course and the hardware used.

We will learn how to compile your CUDA program, how to submit it for execution on one of the compute nodes, and how to examine the results of the execution.

## Code Organization

Lab codes will be typically organized into *lab0.cu*, *Makefile*, and *job.slurm* files:

* *lab0.cu* is where the CUDA code is to be written. In this initial lab the complete code is provided for you to look at, there is no need to modify it. In consecutive labs, you will be writing your code in a similarly named file.

* *Makefile* provides rules to compile and execute your program. Specifically:
    - `make all` will attempt to compile your program
    - `make clean` will remove compiled object files (files with extension .o), but not the executable
    - `make cleanall` will remove all binary/intermediate files, including the compiled executable and results of the execution

* *job.slurm* contains a script for job submission. You should have no reason to modify it.

## Instructions

### To Compile and Execute your program

To compile, simply type `make` or `make all`. This will attempt to compile your code. If there are any errors, you will see a message in the terminal. Fix the errors and try to recompile again. Please note that the compilation takes place on the cluster head node which has all the tools but does not have any GPUs. Therefore, you cannot execute your compiled code on the head node. 

To execute your code, type `sbatch job.slurm`. This will schedule the execution of your program on one of the next available compute nodes. Unlike the head node, compute nodes have GPUs and this is where we want our program to be executed. You will get a message like this `Submitted batch job ID` where the last number is your job ID. Typically, jobs will be executed withing a few seconds. However, if your job is not getting executed immediately, you can check its status by typing `squeue --job ID` (do not forget to replace "ID" number with your actual job ID reported by sbatch). 

Once your program is executed, you will observe the appearance of a new file in your lab directory: `lab0.out`. Make sure to examine this file (`cat lab0.out`) as it contains the output produced during the execution of your program. You need this information, for example, to verify that your program was executed correctly.

For lab0 in particular, the program will output the following information which you will need to complete Lab0 quiz:

* GPU card's name
* GPU computation capabilities
* Maximum number of block dimensions
* Maximum number of grid dimensions
* Maximum size of GPU memory
* Amount of constant and share memory
* Warp size


### To submit lab for grading

There is nothing to submit for grading in this lab, other than the quiz that you need to complet on Canvas. We will show how to submit your code for grading in lab 1.
