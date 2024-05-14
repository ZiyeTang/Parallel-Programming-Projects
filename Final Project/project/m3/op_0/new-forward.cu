#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <sys/time.h>

#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *output, float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil(1.0*Width_out/TILE_WIDTH);

    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;

    if (b < Batch && m < Map_out && h < Height_out && w < Width_out) {
        float acc = 0.0f;
        for (int c = 0; c < Channel; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    if (h+p < Height && w+q < Width) {
                        acc += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
                    }
                }
            }
        }
        out_4d(b, m, h, w) = acc;
    }
    
    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;    

    int num_stream = 10;
    cudaStream_t stream[num_stream];
    for (int i = 0; i < num_stream; i++)
        cudaStreamCreate(&stream[i]);

    float *host_input_pinned, *host_output_pinned;
    cudaMallocHost(&host_input_pinned, Batch * Channel * Height * Width * sizeof(float));
    cudaMallocHost(&host_output_pinned, Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMemcpy(host_input_pinned, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToHost);

    int seg_size = Batch / num_stream;
    int seg_input_size = seg_size * Channel * Height * Width * sizeof(float);
    int seg_output_size = seg_size * Map_out * Height_out * Width_out * sizeof(float);
    int kernel_size = Map_out * Channel * K * K * sizeof(float);


    float *device_input[num_stream];
    float *device_output[num_stream];
    float * device_mask;
    for (int i = 0; i < num_stream; i++) {
        cudaMalloc((void**) &(device_output[i]), seg_output_size);
        cudaMalloc((void**) &(device_input[i]), seg_input_size);
    }
    cudaMalloc((void**) &device_mask, kernel_size);

    int W_grid = ceil(1.0*Width_out/TILE_WIDTH);
    int H_grid = ceil(1.0*Height_out/TILE_WIDTH); 
    int y_dim_grid = H_grid * W_grid;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(seg_size, Map_out, y_dim_grid);

    cudaMemcpyAsync(device_mask, host_mask, kernel_size, cudaMemcpyHostToDevice, stream[0]);
    for (int i=0; i<Batch; i+=seg_size * num_stream) {
        for (int j = 0; j < num_stream; j++) {
            cudaMemcpyAsync(device_input[j], host_input + (i+j*seg_size) * Channel * Height * Width, seg_input_size, cudaMemcpyHostToDevice, stream[j]);
        }  
       
        for (int j = 0; j < num_stream; j++) {
            conv_forward_kernel<<<dimGrid, dimBlock, 0, stream[j]>>>(device_output[j], device_input[j], device_mask, Batch, Map_out, Channel, Height, Width, K);
        }

        for (int j = 0; j < num_stream; j++) {
            cudaMemcpyAsync(host_output_pinned + (i+j*seg_size) * Map_out * Height_out * Width_out, device_output[j], seg_output_size, cudaMemcpyDeviceToHost, stream[j]);
        }
    }
    cudaMemcpy((float*) host_output, host_output_pinned, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyHostToHost);

    for (int i = 0; i < num_stream; i++)
        cudaStreamDestroy(stream[i]);

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFreeHost(host_input_pinned);
    cudaFreeHost(host_output_pinned);

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
    
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    
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