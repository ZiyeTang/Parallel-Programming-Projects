// LAB 2 SP24

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


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Implement matrix multiplication kernel here
  
  int Row = blockIdx.y*blockDim.y+threadIdx.y;
  int Col = blockIdx.x*blockDim.x+threadIdx.x;
  
  if ((Row < numCRows) && (Col < numCColumns)) {
    float Pvalue = 0;
    for (int k = 0; k < numAColumns; ++k)
      Pvalue += A[Row*numAColumns+k] * B[k*numBColumns+Col];
    
    C[Row*numCColumns+Col] = Pvalue;
    // printf("%d %d %f\n", Row, Col, Pvalue);
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
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  
  int sizeA = numARows * numAColumns * sizeof(float);
  int sizeB = numBRows * numBColumns * sizeof(float);
  int sizeC = numCRows * numCColumns * sizeof(float);

  //@@ Allocate the hostC matrix
  
  hostC = (float *)malloc(sizeC);

  //@@ Allocate GPU memory here
  float *dA, *dB, *dC;
  wbCheck(cudaMalloc((void **) &dA, sizeA));
  wbCheck(cudaMalloc((void **) &dB, sizeB));
  wbCheck(cudaMalloc((void **) &dC, sizeC));


  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(dA, hostA, sizeA, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(dB, hostB, sizeB, cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numCColumns/2.0), ceil(numCRows/2.0), 1);
  dim3 dimBlock(2, 2, 1);

  //@@ Launch the GPU Kernel here
  matrixMultiply<<<dimGrid, dimBlock>>>(dA, dB, dC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, dC, sizeC, cudaMemcpyDeviceToHost));

  //@@ Free the GPU memory here
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@Free the hostC matrix
  free(hostC);
  return 0;
}

