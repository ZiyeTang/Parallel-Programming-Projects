// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

#define BLOCK_SIZE 512

//@@ insert code here
__global__ void castToChar(float* input, unsigned char* output, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    output[i] = (unsigned char) (255 * input[i]);
  }  
}


__global__ void colorToGrayscaleConversion(unsigned char * grayImage, unsigned char * rgbImage, int width, int height) {
  int Col = threadIdx.x + blockIdx.x * blockDim.x;
  int Row = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (Col < width && Row < height) {
    int grayOffset = Row*width + Col;

    unsigned char r = rgbImage[3 * grayOffset];
    unsigned char g = rgbImage[3 * grayOffset + 1];
    unsigned char b = rgbImage[3 * grayOffset + 2]; 

    grayImage[grayOffset] = (unsigned char) (0.21f*r + 0.71f*g + 0.07f*b);
  }
}

__global__ void computeHistogram(unsigned char *input, float *histo, int size) {
  __shared__ float private_histo[256];

  if (threadIdx.x < 256)
    private_histo[threadIdx.x] = 0;
  
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  while (i < size) {
    atomicAdd(&(private_histo[input[i]]), 1);
    i += stride;
  }

  __syncthreads();

  if (threadIdx.x < 256)
    atomicAdd(&(histo[threadIdx.x]), private_histo[threadIdx.x]);

}


__global__ void scan(float *input, float *aux, int len) {
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

__global__ void addBlockSum(float* input, float* blockSum,float* output, int len, int size) {
  int index = (blockDim.x * blockIdx.x)*2 + threadIdx.x;
  if(blockIdx.x > 0) {
    if(index < len) {
      output[index] = (input[index] + blockSum[blockIdx.x - 1]) / size;
    }

    if(index+blockDim.x < len) {
      output[index+blockDim.x] = (input[index+blockDim.x] + blockSum[blockIdx.x - 1]) / size;
    }
  } else {
    if(index < len) {
      output[index] = input[index] / size;
    }

    if(index+blockDim.x < len) {
      output[index+blockDim.x] = input[index+blockDim.x] / size;
    }
  }
  
}


__device__ float clamp(float x, float start, float end) {
  return min(max(x, start),end);
}

__device__ float correct_color(unsigned char val, float* cdf) {
  return clamp(255.0*(cdf[val] - cdf[0])/(1.0 - cdf[0]), 0.0, 255.0);
}
	

__global__ void histogramEqualization(unsigned char* ucharImage, float* cdf, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    ucharImage[i] = correct_color(ucharImage[i], cdf);
  }
}

__global__ void castToFloat(unsigned char* input, float* output, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
    output[i] = (float) (1.0 * input[i] / 255.0);
  }  
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);


  //@@ insert code here
  int image_size = imageWidth * imageHeight * imageChannels;
  int numBlocks = ceil(1.0 *  HISTOGRAM_LENGTH / (BLOCK_SIZE*2));

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = (float *)malloc(image_size * sizeof(float));
  

  float* deviceInput;
  float* deviceOutput;
  unsigned char * deviceCharImage;
  unsigned char * deviceGrayImage;
  float *deviceHisto;
  float *deviceCDF;
  float *deviceAux;

  cudaMalloc((void **)&deviceInput, image_size * sizeof(float));
  cudaMalloc((void **)&deviceOutput, image_size * sizeof(float));
  cudaMalloc((void **)&deviceCharImage, image_size * sizeof(unsigned char));
  cudaMalloc((void **)&deviceGrayImage, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&deviceHisto, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **)&deviceAux, numBlocks * sizeof(float));

  cudaMemcpy(deviceInput, hostInputImageData, image_size * sizeof(float), cudaMemcpyHostToDevice);
  
  
  dim3 DimGridCast(ceil(1.0*image_size/256), 1, 1);
  dim3 DimBlockCast(256, 1, 1);

  dim3 DimGridGrayScale(ceil(1.0*imageWidth/16), ceil(1.0*imageHeight/16), 1);
  dim3 DimBlockGrayScale(16, 16, 1);

  dim3 DimGridHisto(ceil(1.0 * imageWidth * imageHeight / 4.0 / HISTOGRAM_LENGTH), 1, 1);
  dim3 DimBlockHisto(HISTOGRAM_LENGTH, 1, 1);
  
  dim3 DimGridScan(numBlocks, 1, 1);
  dim3 DimGridScanBlockSum(1, 1, 1);
  dim3 DimBlockScan(BLOCK_SIZE, 1, 1);

  //Cast to unsigned char
  castToChar<<<DimGridCast, DimBlockCast>>>(deviceInput, deviceCharImage, image_size);
  cudaDeviceSynchronize();


  //Convert to grayscale
  colorToGrayscaleConversion<<<DimGridGrayScale, DimBlockGrayScale>>>(deviceGrayImage, deviceCharImage, imageWidth, imageHeight);
  cudaDeviceSynchronize();


  //Compute histogram
  computeHistogram<<<DimGridHisto, DimBlockHisto>>>(deviceGrayImage, deviceHisto, imageWidth * imageHeight);
  cudaDeviceSynchronize();

  

  //Compute CDF
  scan<<<DimGridScan, DimBlockScan>>>(deviceHisto, deviceAux, HISTOGRAM_LENGTH);
  cudaDeviceSynchronize();

  scan<<<DimGridScanBlockSum, DimBlockScan>>>(deviceAux, NULL, numBlocks);
  cudaDeviceSynchronize();

  addBlockSum<<<DimGridScan, DimBlockScan>>>(deviceHisto, deviceAux, deviceCDF, HISTOGRAM_LENGTH, imageWidth * imageHeight);
  cudaDeviceSynchronize();



  //Apply the histogram equalization function
  histogramEqualization<<<DimGridCast, DimBlockCast>>>(deviceCharImage, deviceCDF, image_size);
  cudaDeviceSynchronize();

  castToFloat<<<DimGridCast, DimBlockCast>>>(deviceCharImage, deviceOutput, image_size);
  cudaDeviceSynchronize();


  cudaMemcpy(hostOutputImageData, deviceOutput, image_size * sizeof(float), cudaMemcpyDeviceToHost);
  wbImage_setData(outputImage, hostOutputImageData);

  cudaFree(deviceInput);
  cudaFree(deviceCharImage);
  cudaFree(deviceGrayImage);
  cudaFree(deviceHisto);
  cudaFree(deviceAux);
  cudaFree(deviceCDF);
  cudaFree(deviceOutput);
  wbSolution(args, outputImage);

  //@@ insert code here
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}

