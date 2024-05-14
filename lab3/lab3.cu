#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileM [TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN [TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = TILE_WIDTH * by + ty;
  int Col = TILE_WIDTH * bx + tx;
  float pvalue = 0;

  for(int q = 0; q < (numAColumns - 1) / TILE_WIDTH + 1; ++q) {
    if (Row < numARows && q*TILE_WIDTH+tx < numAColumns) {
      subTileM[ty][tx] = A[Row*numAColumns + q*TILE_WIDTH+tx];
    } else{
      subTileM[ty][tx] = 0;
    }

    if (q*TILE_WIDTH+ty < numBRows && Col < numBColumns) {
      subTileN[ty][tx] = B[(q*TILE_WIDTH+ty)*numBColumns + Col];
    } else{
      subTileN[ty][tx] = 0;
    }
    
    __syncthreads();
    if (Row < numCRows && Col < numCColumns) {
      for (int k = 0; k < TILE_WIDTH; ++k)
        pvalue += subTileM[ty][k] * subTileN[k][tx];
    }

    
    __syncthreads();
  }

  if (Row < numCRows && Col < numCColumns){
    C[Row*numCColumns + Col] = pvalue;
  }
    
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  // wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  // wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  int sizeA = numARows * numAColumns * sizeof(float);
  int sizeB = numBRows * numBColumns * sizeof(float);
  int sizeC = numCRows * numCColumns * sizeof(float);

  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(sizeC);

  //@@ Allocate GPU memory here
  float *dA, *dB, *dC;
  cudaMalloc((void**) &dA, sizeA); 
  cudaMalloc((void**) &dB, sizeB); 
  cudaMalloc((void**) &dC, sizeC); 

  //@@ Copy memory to the GPU here
  cudaMemcpy(dA, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hostB, sizeB, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil((1.0*numCColumns) / TILE_WIDTH ), ceil((1.0*numCRows) / TILE_WIDTH), 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1); 

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>> (dA, dB, dC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, dC, sizeC, cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix
  free(hostC);
  return 0;
}
