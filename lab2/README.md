# Basic Matrix Multiplication

## Objective

The purpose of this lab is to implement a basic dense matrix multiplication routine.

### Retrieving Assignments ###

To retrieve (or update) released assignments, go to your `ece408git` folder and run the following:

* `git fetch release`
* `git merge release/main -m "some comment" --allow-unrelated-histories`
* `git push origin main`

where "some comment" is a comment for your submission. The last command pushes the newly merged files to your remote repository. If something ever happens to your repository and you need to go back in time, you will be able to revert your repository to when you first retrieved an assignment.

One more thing, if you are asked to enter your git credentials (PAT) each time you try to pull or push, you can try to store your git credentials once and for all: just type `git config credential.helper store` in your Delta terminal.

## Instructions

Edit the code in `lab2.cu` to perform the following:

* Implement the CUDA kernel
* Allocate device memory
* Copy host memory to device
* Initialize thread block and kernel grid dimensions
* Invoke CUDA kernel
* Copy results from device to host
* Free device memory


Instructions about where to place each part of the code are demarcated by the `//@@` comment lines.

Refer to the instructions in lab0 for how to compile and run the code. One noticeable difference is that `job.slurm` now defines separate files for stderr and stdout streams; they are directed to `lab2.err` and `lab2.out` files, respectively. Therefore, you should check both files for messages instead of just `lab2.out` file.

## Submission

Every time you want to submit the work, you will need to `add`, `commit`, and `push` your work to your git repository. This can always be done using the following commands on a command line while within your ECE 408 directory:

* ```git add -u```
* ```git commit -m "some comment"```
* ```git push origin main```


## Suggestions (for all labs)  

* Do not modify the template code provided -- only insert code where the `//@@` demarcation is placed.  
* Develop your solution incrementally and test each version thoroughly before moving on to the next version.  
* If you get stuck with boundary conditions, grab a pen and paper. It is much easier to figure out the boundary conditions there.  
* Implement the serial CPU version first, this will give you an understanding of the loops.  
* Get the first dataset working first. The datasets are ordered so the first one is the easiest to handle.  
* Make sure that your algorithm handles non-regular dimensional inputs (not square or multiples of 2). The slides may present the algorithm with nice inputs, since it minimizes the conditions. The datasets reflect different sizes of input that you are expected to handle.  
* Make sure that you test your program using all the datasets provided. `job.slurm` file contains the code that runs your implementation with all datasets. You can modify it to run one dataset at a time.   
* Check for errors: for example, when developing CUDA code, one can check for if the function call succeeded and print an error if not via the following macro:
```
#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
```

An example usage is ```wbCheck(cudaMalloc(...))```.
