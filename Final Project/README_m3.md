# ECE408/CS483 Final Project

## Milestone 3: GPU Convolution Kernel Optimizations

***Deadline: April 26, 2024, 8 PM CST***

| Step | Deliverables |
| ------------ | ------------ |
| 1 | Implement multiple GPU optimizations |
| 2 | Acheive op times (sum) of <= **65ms** |
| 3 | Write your report and upload PDF to Canvas(https://canvas.illinois.edu/courses/43562/quizzes/303269) |
| 4 | Submit your work for grading! |

### Create your own GPU optimizations! The real race against time.

Your goal is to implement streams and at least **10 other points** of GPU optimizations (as seen in [optimizations](#optimizations)), additional optimization points will count towards extra credits. Optimizations are up to you to pick from and can be implemented separately from each other and/or stack each optimization in order to maximize performance.

**You MUST implement optimization number 0 (Streams). This is will not count towards your 10 points.**

You will be performing analysis on every optimization. You must clarify which baseline is used for any given optimization. Any analysis on individual optimization should be compared against your milestone 2 baseline. If you begin to stack multiple optimizations, each new optimization should be compared against the previous version without said optimization. This adds more context and proper meaning to your analysis.

Op times are sometimes quite arbritrary and is not the only way to show improvement in your kernel. It is fine if an optimization is not improving the performance against the baseline,
but you have to provide your implementation in your code and sufficient profiling results in your report. Also please remember when profiling your optimizations, replace the `#SBATCH --constraint="scratch"` with `#SBATCH --constraint="scratch,perf,nvperf"` flag to run your code.

If you have done milestone 2 correctly, for a batch size of 10000, the sum between the first and second layer OP Times should equal about **120ms**.

In order to achieve full credit for the performace in milestone 3, your final submission must bring down the sum of the op times to **65ms** or less for a batch size of 10000. Any submissions between **65ms** and **120ms** will be given a performance grade linearly extrapolated from the performance relative to these two values. 

Any submission slower than **120ms** will recieve no credit for the performance grade.

We only grade the performance on one optimization inside `new-forward.cu` of final submission, other optimizations will be graded depends on your report.

**IMPORTANT:All your gpu kernel calls need to take place inside conv_forward_gpu() for final submission**

### Performance Analysis using Nsight-Systems and Nsight-Compute

Use the NVIDIA Nsight-Systems (`nsys`) and Nsight-Compute (`nv-nsight-cu-cli`) and your analysis information to describe the effect that your optimizations had on the performance of your convolution.
If possible, you should try to separate the effect of each optimization in your analysis. 

Please ensure that your submission includes both binary files for profiling (Nsight-Systems and Nsight-Compute) for each of your optimization. More in information is below.

## Documentation on running your code

Please **do not** run your code with `#SBATCH --constraint="scratch,perf,nvperf"` **if you are not actively profiling your code**. Not only will it take longer for you to run your code, it slows traffic for everyone. For basic functionality/optime check, please use the flag `#SBATCH --constraint="scratch"`. This flag can be found in `./Project/m3.slurm`.

To compile your code (do this for every change you make):
- `./run.sh build`

To run your code:
- `sbatch m3.slurm`

Checking your output:
- `m3.out` has your outputs
- `Milestone3.err` has your error outputs. Check here for seg-faults or anything similar.

Reminder: this will **only** run your `new-forward.cu` file inside of `/project/src/layer/custom/`.

## Submission Guidelines


### Project Files Submission Guideline through Github
- Your **final** submission (stacked optimizations or not) should be your `/project/src/layer/custom/new-forward.cu` file. 
    - This is your final submission that we will test for a combined optime of <= 65ms. 
    - **Though streams is mandatory as part of your optimizations, your final submission for performance test must be done on a single stream.**
- Look under `./project/m3` and find the optimization folders.
- **Each** optimization you implemeneted should have each own folder with the following requirements:
    - name of the folder should have the following format: `op_#`. (see the optimization numbers in [optimizations](#optimizations))
    - it should contain an non-stacked version of your implementation
        - a functional copy of `new-forward.cu` with **ONLY** this implementation added on from the base m2 implementation
        - we will perform functionality checks on every individual optimization
    - all profiling results (your outputted binary analysis files) that you included in your final report.
    - feel free to add more folders if needed. 
- **You must have a folder for each optimization individually** even if you stacked all of them for your final submission.
``` 
|--->Project
    |--->project  
        |---> /m3
            |---> /op_0
                |---> new-forward.cu
                |---> m3.out
                |---> analysis.ncu-rep
                |---> output
                |---> analysis.nsys-rep(optional)
                |--->...( other useful profiling results)
            |---> /op_1
            |---> /op_4
            |---> /op_12
            ...
        ...
    ...
```

- Push your code to GitHub!
    - Only add your changes from `/project/m3/` and `/project/src/layer/custom/new-forward.cu`
- **We strongly recommend that you periodically make commits**, local or not, to ensure that you have a record of your work saved. You can always just soft reset to merge your commits. It also provides proof in case something goes wrong with submissions.

### Milestone 3 Report Submission Guidelines through Canvas
As the world's best engineers and scientists, it is imperative to document our work meticulously and analyze data with scientific rigor. In case analyze statistical results from your profiling results, we recommend to take a look at this [thesis](http://impact.crhc.illinois.edu/shared/report/phd-thesis-shane-ryoo.pdf) and pay particular attention to Section 5.1 for reference and inspiration.

**We give you a report template: `ECE408_S24_netid_m3_report.docx`.** Please use this document to get started with your report.

Follow the following steps for each GPU optimization:
| Step | For each optimization |
| ------------ | ------------ |
| 1 | Name the optimization and corresponding number |
| 2 | How does the optimization work?  How does this method theoretically optimize your convolution kernel? Expected behavior? |
| 3 | How did you implement your code? You must thoroughly explain the thought process behind your implementation with code snippets and justify the correctness of your implementation with proper profiling results. |
| 4 | Did the performance match your expection? Analyse the profiling results as a scientist. |
| 5 | Does the optimization synergize with any of other optimizations? How? |
| 6 | What references did you use when implementing this technique? |

When Submitting:
- be sure to include any external references used during identification or development of the optimization.
- export your report as pdf and name it as `ECE408_S24_netid_m3_report.pdf`
- **upload to Canvas(https://canvas.illinois.edu/courses/43562/quizzes/303269)**

## Optimizations

These are the list of optimizations we will consider valid for Milestone 3. To obtain full credit for Milestone 3, you must implement a minimum of 14 points of optimizations, including the mandatory optimization 0. If you would like to implement a potential optimization that is not on this list, please consult a TA or instructor beforehand to verify that the optimization is valid and to assign it a point value. We'd love to hear about your creative ideas!

| Number | Optimization | Points |
| ------------ | ------------ | ------------ |
| **0** | **Using Streams to overlap computation with data transfer (required)** | **4** |
| 1 | Tiled (shared memory) convolution | 2 |
| 2 | Input matrix unrolling & tiled matrix multiplication using shared memory | 3 |
| 3 | Kernel fusion for unrolling and matrix-multiplication (requires op #2) | 2 |
| 4 | Weight matrix (Kernel) in constant memory | 1 |
| 5 | Tuning with restrict and loop unrolling | 2 |
| 6 | Sweeping various parameters to find best values (block sizes, amount of thread coarsening) (requires op #2) -- requires tables/graphs in Report| 2 |
| 7 | Multiple kernel implementations for different layer sizes | 1 |
| 8 | Input channel reduction: tree | 2 |
| 9 | Input channel reduction: atomics | 1 |
| 10 | Fixed point (FP16) arithmetic implementation (this can modify model accuracy slightly) | 3 |
| 11 | Using Tensor Cores to speed up matrix multiplication | 5 |
| 12 | An advanced matrix multiplication algorithm (register-tiled, for example) | 6 |
| 13 | Overlap-Add method for FFT-based convolution (note this is **very** hard, and may not yield a large performace increase due to mask size) | 8 |


### Extra credits in the project

Make sure you implement 14 optimization points including streams for this milestone first before considering extra credits. If you implement some optimizations incorrectly or you didn't include enough information in your report, we will not consider extra points as part of your project until you have accumulated 14 correct optimization points. Additional optimization points will count towards extra credits. Each additional optimization point is worth 2%. You can earn 10% maximum towards your project grade.

## Rubric

The overall project score will be computed as follows. We will release rubric details of later milestones based on the class schedule.
So please always do `git pull` to update the project instructions.

1. Milestone 1 ( 20% )
    - Correctness ( 15% )
    - Report ( 5% )
2. Milestone 2 ( 30% )
    - Correctness ( 20% )
    - Report( 10% )
3. Milestone 3 ( 50% )
    - Overall Performance ( 8% )
        - **ALL** your gpu kernel calls need to be launched inside conv_forward_gpu() for your performance submission
        - Though streams is mandatory as part of your optimizations, **your final submission for performance test must be done on a single stream.**
    - Report completeness and optimization correctness ( 42% )
        - Streams ( 12% )
        - Other 10 optimization points ( 3% per point, 30% in total )
5. Extra Credit ( up to + 10 maximum, +2% per additional optimization point )

## Appendix

### Skeleton Code Description
`project/src/layer/custom/cpu-new-forward.cc` and `project/src/layer/custom/new-forward.cu` containes skeleton implementations for the CPU and GPU convolutions respectively. You can complete the project by modifying these two files only. `project/src/layer/custom/cpu-new-forward.h` and `project/src/layer/custom/gpu-new-forward.h` are the respective header files. You need not modify these files unless you need to declare your own functions.


### Checking for Errors

Within `project/src/layer/custom/new-forward.cu`, you can use the predefined error handling code to catch CUDA errors or, you can define a macro/function similar to `wbCheck` used in WebGPU.

To catch memory errors, prepend your command with `cuda-memcheck`. 
Assume we want to check memory errors on Milestone3 binary, 
in your `m3.slurm`, run 

    - /bin/bash -c "cuda-memcheck ./m3"

### Profiling

You can gather system level performance information using `nsys`.

For detailed kernel level GPU profiling, use `nv-nsight-cu-cli` and view that information with `nv-nsight-cu`. To enable profiling with these tools,
you have to remove `-DCMAKE_CXX_FLAGS=-pg` in cmake and make line of your `run.sh`:

    cmake ./project/ && make -j8

You can see some simple information like so (as we did in milestone 2):

    srun nsys profile --stats=true <your command here>

You can additionally gather some basic kernel level performance metrics.

    srun nv-nsight-cu-cli -f -o analysis_file <your command here>

This will generate `analysis_file.ncu-rep` with a very basic profiling result.
You can gather more detailed kernel level performance metrics.

    ncu --set full -o analysis_file <your command here>
    
This will generate `analysis_file.ncu-rep` with a detailed profiling result. But the OP time may be very big in this case, we won't use this as a final execution time.

You can use the NVIDIA Nsight Compute GUI (`nv-nsight-cu`) to import those files.
You will need to install NVIDIA NSight Compute on your own machine. It can be downloaded as a standalone application. See instructions [here](#nsight-compute-installation)

To import the files:
* Launch the GUI `/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu` (or from wherever you installed it)
* Close the intial Quick Launch menu
* Go to File > Open File and select the `.ncu-rep` file downloaded from your Delta platform

*OR*
* Directly launch from the terminal `/usr/local/NVIDIA-Nsight-Compute/nv-nsight-cu <filename>.ncu-rep`

For a high-level overview of the Nsight software, visit [here](https://developer.nvidia.com/tools-overview).

### Nsight-compute Installation

Nsight-Compute can be installed as a standalone application. You do not need CUDA to be installed. You can download the installer from NVIDIA's [website](https://developer.nvidia.com/gameworksdownload#?dn=nsight-compute-2020-3-0)

## License

NCSA/UIUC © 2020 [Carl Pearson](https://cwpearson.github.io)

## Contributors

* [Carl Pearson](https://cwpearson.github.io)
* [Vikram Mailthody](https://github.com/msharmavikram/)
* Andrew Schuh
* Abdul Dakkak
* Zaid Qureshi
* Rui Lan
* Zhicun Wan
* Ben Schreiber
* James Cyriac
* Jonathan Nativ
* Shuangliang Chen
* Huili Tao
* Howie Liu
* Thomas Bae
