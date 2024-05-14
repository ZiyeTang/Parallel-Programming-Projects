#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define KERNEL_WIDTH 3
#define TILE_WIDTH 8

//@@ Define constant memory for device kernel here
__constant__ float dk[KERNEL_WIDTH][KERNEL_WIDTH][KERNEL_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int x_o = blockIdx.x * TILE_WIDTH + tx;
  int y_o = blockIdx.y * TILE_WIDTH + ty;
  int z_o = blockIdx.z * TILE_WIDTH + tz;
  
  int x_i = x_o - (KERNEL_WIDTH / 2);
  int y_i = y_o - (KERNEL_WIDTH / 2);
  int z_i = z_o - (KERNEL_WIDTH / 2);

  __shared__ float tile[TILE_WIDTH + KERNEL_WIDTH - 1][TILE_WIDTH + KERNEL_WIDTH - 1][TILE_WIDTH + KERNEL_WIDTH - 1];

  
  if ((x_i >= 0) && (x_i < x_size) && (y_i >= 0) && (y_i < y_size) && (z_i >= 0) && (z_i < z_size)) {
    tile[tz][ty][tx] = input[z_i * y_size * x_size + y_i * x_size + x_i];
  } else {
    tile[tz][ty][tx] = 0.0f;
  }
  __syncthreads ();

  float Pvalue = 0.0f;
  if (tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH && x_o < x_size && y_o < y_size && z_o < z_size) {
    for(int i = 0; i < KERNEL_WIDTH; i++) {
      for(int j = 0; j < KERNEL_WIDTH; j++) {
        for (int k = 0; k < KERNEL_WIDTH; k++) {
          Pvalue += dk[i][j][k] * tile[i+tz][j+ty][k+tx];
        }
      }
    }

    output[z_o * y_size * x_size + y_o * x_size + x_o] = Pvalue;
  }
  
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *din, *dout;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int io_size = (inputLength - 3) * sizeof(float);
  cudaMalloc((void **) &din, io_size); 
  cudaMalloc((void **) &dout, io_size);
  

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(din, &hostInput[3], io_size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dk, hostKernel, kernelLength*sizeof(float));


  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil((1.0 * x_size) / TILE_WIDTH), ceil((1.0 * y_size) / TILE_WIDTH), ceil((1.0 * z_size) / TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH + KERNEL_WIDTH - 1, TILE_WIDTH + KERNEL_WIDTH - 1, TILE_WIDTH + KERNEL_WIDTH - 1);

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(din, dout, z_size,  y_size, x_size);
  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], dout, io_size, cudaMemcpyDeviceToHost);



  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(din);
  cudaFree(dout);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

