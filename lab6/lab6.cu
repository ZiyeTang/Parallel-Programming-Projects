// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *aux, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float tile[2*BLOCK_SIZE];
  int index = (blockDim.x * blockIdx.x) * 2 + threadIdx.x;
  
  if (index < len) {
    tile[threadIdx.x] = input[index];
  } else {
    tile[threadIdx.x] = 0;
  }

  if (index + blockDim.x < len) {
    tile[threadIdx.x + blockDim.x] = input[index + blockDim.x];
  } else {
    tile[threadIdx.x + blockDim.x] = 0;
  }

  
  int stride = 1;
  while(stride < 2*BLOCK_SIZE) {
    __syncthreads();
    int cur_idx = (threadIdx.x+1)*stride*2 - 1;
    if(cur_idx < 2*BLOCK_SIZE && (cur_idx-stride) >= 0)
      tile[cur_idx] += tile[cur_idx-stride];
    stride = stride*2;
  }

  stride = BLOCK_SIZE/2;
  while(stride > 0) {
    __syncthreads();
    int cur_idx = (threadIdx.x+1)*stride*2 - 1;
    if ((cur_idx+stride) < 2*BLOCK_SIZE)
      tile[cur_idx+stride] += tile[cur_idx];
    stride = stride / 2;
  }

  __syncthreads();
  
  if (index < len) {
    input[index] = tile[threadIdx.x];
  }

  if (index + blockDim.x < len) {
    input[index + blockDim.x] = tile[threadIdx.x + blockDim.x];
  }

  if (aux != NULL) {
    aux[blockIdx.x] = tile[2*BLOCK_SIZE - 1];
  }
}

__global__ void addBlockSum(float* input, float* blockSum, float* output, int len) {
  int index = (blockDim.x * blockIdx.x)*2 + threadIdx.x;
  if(blockIdx.x > 0) {
    if(index < len) {
      output[index] = input[index] + blockSum[blockIdx.x - 1];
    }

    if(index+blockDim.x < len) {
      output[index+blockDim.x] = input[index+blockDim.x] + blockSum[blockIdx.x - 1];
    }
  } else {
    if(index < len) {
      output[index] = input[index];
    }

    if(index+blockDim.x < len) {
      output[index+blockDim.x] = input[index+blockDim.x];
    }
  }
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceAux;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));

  int numBlocks = ceil(1.0 * numElements / (BLOCK_SIZE*2));
  wbCheck(cudaMalloc((void **)&deviceAux, numBlocks * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(numBlocks, 1, 1);
  dim3 DimGridScanBlockSum(1, 1, 1);
  // dim3 DimGridAddBlockSum(ceil(1.0 * numElements / BLOCK_SIZE), 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceAux, numElements);
  scan<<<DimGridScanBlockSum, DimBlock>>>(deviceAux, NULL, numBlocks);
  addBlockSum<<<DimGrid, DimBlock>>>(deviceInput, deviceAux, deviceOutput, numElements);

  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAux);
  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);
  return 0;
}

