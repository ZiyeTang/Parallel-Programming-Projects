# Vector Addition

## Objective

The purpose of this lab is for you to become familiar with using the CUDA API by implementing a simple vector addition kernel and its associated host code as shown in the lectures.

### Retrieving Assignments ###

To retrieve (or update) released assignments, go to your ece408git folder and run the following:

* `git fetch release`

* `git merge release/main -m "some comment" --allow-unrelated-histories`

* `git push origin main`

where "some comment" is a comment for your submission. The last command pushes the newly merged files to your remote repository. If something ever happens to your repository and you need to go back in time, you will be able to revert your repository to when you first retrieved an assignment.

## Instructions

Edit the code in `lab1.cu` to perform the following:

* Allocate device memory
* Copy host memory to device
* Initialize thread block and kernel grid dimensions
* Invoke CUDA kernel
* Copy results from device to host
* Free device memory
* Implement the CUDA kernel

Instructions about where to place each part of the code is demarcated by the //@@ comment lines.

Refere to the instructions in lab0 for how to compile and run the code.

## Submission

Every time you want to save the work, you will need to add, commit, and push your work to your git repository. This can always be done using the following commands on a command line while within your ECE 408 directory:

* ```git add -u```
* ```git commit -m "REPLACE THIS WITH YOUR COMMIT MESSAGE"```
* ```git push origin main```
