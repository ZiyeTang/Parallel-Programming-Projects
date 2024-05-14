// LAB 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<len) out[i] = in1[i] + in2[i];
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  wbLog(TRACE, "The input length is ", inputLength);

  //@@ Allocate GPU memory here
  int size = inputLength * sizeof(float);
  float *i1, *i2, *out;
  cudaMalloc((void **) &i1, size);
  cudaMalloc((void **) &i2, size);
  cudaMalloc((void **) &out, size);
  

  //@@ Copy memory to the GPU here
  cudaMemcpy(i1, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(i2, hostInput2, size, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(inputLength / 256, 1, 1);
  if (0 != (inputLength % 256)) { DimGrid.x++; }
  dim3 DimBlock(256, 1, 1);
  

  //@@ Launch the GPU Kernel here to perform CUDA computation
  vecAdd<<<DimGrid,DimBlock>>>(i1, i2, out, inputLength);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, out, size, cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here

  cudaFree(i1);
  cudaFree(i2);
  cudaFree(out);

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
