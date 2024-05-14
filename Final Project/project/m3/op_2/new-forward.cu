#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define TILE_WIDTH2 16

__global__ void conv_forward_kernel(const float *A, const float *B, float *C, const int numARows, const int numAColumns, const int numBColumns, const int start)
{
    __shared__ float subTileM [TILE_WIDTH2][TILE_WIDTH2];
    __shared__ float subTileN [TILE_WIDTH2][TILE_WIDTH2];

    int b = blockIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = TILE_WIDTH2 * by + ty;
    int Col = TILE_WIDTH2 * bx + tx;
    float pvalue = 0;

    for(int q = 0; q < (numAColumns - 1) / TILE_WIDTH2 + 1; ++q) {
        if (Row < numARows && q*TILE_WIDTH2 + tx < numAColumns) {
            subTileM[ty][tx] = A[Row*numAColumns + q*TILE_WIDTH2 + tx];
        } else{
            subTileM[ty][tx] = 0;
        }

        if (q*TILE_WIDTH2 + ty < numAColumns && Col < numBColumns) {
            subTileN[ty][tx] = B[b*numAColumns*numBColumns + (q*TILE_WIDTH2 + ty)*numBColumns + Col];
        } else{
            subTileN[ty][tx] = 0;
        }

        __syncthreads();
        if (Row < numARows && Col < numBColumns) {
            for (int k = 0; k < TILE_WIDTH2; ++k)
                pvalue += subTileM[ty][k] * subTileN[k][tx];
        }


        __syncthreads();
    }

    if (Row < numARows && Col < numBColumns){
        C[(b+start)*numARows*numBColumns + Row*numBColumns + Col] = pvalue;
    }
}


__global__ void unroll_kernel(float *output, const float *input, const int Channel, const int Height, const int Width, const int K, const int start)
{

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_3d(i2, i1, i0) output[(i2) * (K* K * Channel * Height_out * Width_out) + (i1) * (Height_out * Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    int b = blockIdx.z;
    int w = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int h = blockIdx.y * TILE_WIDTH + threadIdx.y;
    

    if (h < Height_out && w < Width_out) {
        for (int c = 0; c < Channel; c++) {
            int w_base = c * (K*K);
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    out_3d(b, w_base + p * K + q, h * Width_out + w) = in_4d(b+start, c, h + p, w + q);
                }
            }
        }
    }
    
    #undef out_3d
    #undef in_4d
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    // Allocate memory and copy over the relevant data structures to the GPU
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    
    int output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    int input_size = Batch * Channel * Height * Width * sizeof(float);
    int kernel_size = Map_out * Channel * K * K * sizeof(float);

    cudaMalloc((void**) device_output_ptr, output_size);
    cudaMalloc((void**) device_input_ptr, input_size);
    cudaMalloc((void**) device_mask_ptr, kernel_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, kernel_size, cudaMemcpyHostToDevice);
    

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int W_grid = ceil(1.0*Width_out/TILE_WIDTH);
    int H_grid = ceil(1.0*Height_out/TILE_WIDTH); 

    float* unrolled_matrix;
    int numRow = K * K * Channel;
    int numCol = Height_out * Width_out;
    
    int seg_size = min(Batch, 5000);
    dim3 dimBlock_unroll(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid_unroll(W_grid, H_grid, seg_size);
    dim3 dimBlock_forward(TILE_WIDTH2, TILE_WIDTH2, 1);
    dim3 dimGrid_forward(ceil(1.0*numCol/TILE_WIDTH2), ceil(1.0*Map_out/TILE_WIDTH2), seg_size);
    
    cudaMalloc((void**) &unrolled_matrix,  seg_size * numRow * numCol * sizeof(float));
    for (int i = 0; i < Batch; i+=seg_size) {
        unroll_kernel<<<dimGrid_unroll, dimBlock_unroll>>>(unrolled_matrix, device_input, Channel, Height, Width, K, i);
        conv_forward_kernel<<<dimGrid_forward, dimBlock_forward>>>(device_mask, unrolled_matrix, device_output, Map_out, numRow, numCol, i);
    }
    cudaFree(unrolled_matrix);
    
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}