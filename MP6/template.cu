// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define IMAGE_BLOCK_SIZE 256
#define RGB_BLOCK_SIZE 16
#define BLOCK_SIZE 16

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ insert code here

/*
  Description: This kernel will divide an image by blocks of size 256 where each thread in a block will
  do the casting
*/
__global__ void floatToUnsignedChar (float * inputImage, unsigned char * ucharImage, int numElements) {

  int tx = threadIdx.x;
  int bx = blockIdx.x;

  int start_idx = bx * blockDim.x + tx;

  if (start_idx < numElements)
    ucharImage[start_idx] = (unsigned char) (255.0 * inputImage[start_idx]);
}

__global__ void unsignedCharToFloat (unsigned char * ucharImage, float * outputImage, int numElements) {

  int tx = threadIdx.x;
  int bx = blockIdx.x;

  int start_idx = bx * IMAGE_BLOCK_SIZE + tx;

  if (start_idx < numElements)
    outputImage[start_idx] = (float) (ucharImage[start_idx] / 255.0);
}

__global__ void RGBToGrayScale (unsigned char * inputImage, unsigned char * grayImage, int Height, int Width) {

  __shared__ unsigned char sharedImage[RGB_BLOCK_SIZE * RGB_BLOCK_SIZE * 3];

  int tx = threadIdx.x; // thread's x position in the tile's width
  int ty = threadIdx.y; // thread's y position in the tile's height
  int bx = blockIdx.x; // tile's start position according to the image width
  int by = blockIdx.y; // tile's start position according the the image height
  int image_x_pos = bx * blockDim.x + tx;
  int image_y_pos = by * blockDim.y + ty;
  unsigned char r;
  unsigned char g;
  unsigned char b;

  /* i2 == row, i1 == col, i0 == channel */
  #define shared_image(i2, i1, i0) sharedImage[(i2) * (RGB_BLOCK_SIZE) * 3 + (i1) * 3 + (i0)]
  #define input_image(i2, i1, i0) inputImage[(i2) * (Width) * 3 + (i1) * 3 + (i0)]
  #define gray_image(i1, i0) grayImage[(i1) * (Width) + (i0)]

  /* use threads to load in the data */
  if ((image_x_pos < Height) && (image_y_pos < Width)) {
    for (int i = 0; i < 3; i++) {
      shared_image(ty, tx, i) = input_image(image_y_pos, image_x_pos, i);
    }
  }
  __syncthreads();

  /* grab RGB values */
  r = shared_image(ty, tx, 0);
  g = shared_image(ty, tx, 1);
  b = shared_image(ty, tx, 2);

  /* send them to gray scale image */
  if (image_y_pos < Height && image_x_pos < Width)
    gray_image(image_y_pos, image_x_pos) = (unsigned char) (0.21*r + 0.71*g + 0.07*b);

  #undef shared_image
  #undef input_image
  #undef gray_image
}

__global__ void histrogram_privatized_kernel (unsigned char * inputImage, unsigned int * histogram, int Height, int Width) {
  int tx = threadIdx.x; // thread's x position in the tile's width
  int ty = threadIdx.y; // thread's y position in the tile's height
  int bx = blockIdx.x; // tile's start position according to the image width
  int by = blockIdx.y; // tile's start position according the the image height
  int image_x_pos = bx * blockDim.x + tx;
  int image_y_pos = by * blockDim.y + ty;
  int histogram_pos = ty * blockDim.x + tx;

  __shared__ unsigned int histo_s[HISTOGRAM_LENGTH];

  #define gray_image(i1, i0) inputImage[(i1) * (Width) + (i0)]

  /* initialize private histrogram */
  if (histogram_pos < HISTOGRAM_LENGTH)
    histo_s[histogram_pos] = 0;

  __syncthreads();

  /* check that the thread is a valid position in the image */
  if (image_x_pos < Width && image_y_pos < Height) {
    atomicAdd(&(histo_s[gray_image(image_y_pos, image_x_pos)]), 1);
  }

  __syncthreads();

  /* atomically add private histogram to the actual histrogram */
  if (histogram_pos < HISTOGRAM_LENGTH)
    atomicAdd(&(histogram[histogram_pos]), histo_s[histogram_pos] / (Height*Width));

  #undef gray_image
}

__global__ void scan(unsigned int *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int input_idx = tx + (2*BLOCK_SIZE * bx);

  /* load shared memory */
  T[tx] = input_idx < len ? input[input_idx] : 0;
  T[tx + blockDim.x] = input_idx + blockDim.x < len ? input[input_idx + blockDim.x] : 0;

  /* prescan step */
  int stride = 1;
  while (stride < 2 * BLOCK_SIZE) {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if (index < 2*BLOCK_SIZE && (index - stride) >= 0)
      T[index] += T[index-stride];
    stride *= 2;
  }

  /* postscan step */
  stride = BLOCK_SIZE/2;
  while (stride > 0) {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if ((index + stride) < 2 * BLOCK_SIZE)
      T[index + stride] += T[index];
    stride = stride / 2;
  }

  /* throw data back into output */
  __syncthreads();
  if (input_idx < len)
    output[input_idx] = T[tx];
  if (input_idx + blockDim.x < len)
    output[input_idx + blockDim.x] = T[tx + blockDim.x];

}

__global__ void colorCorrection(float * cdf, unsigned char * image, int Height, int Width) {

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int cdfmin = cdf[0];
  int x, start, end;
  start = 0;
  end = 255.0;

  int image_x_pos = bx * blockDim.x + tx;
  int image_y_pos = by * blockDim.y + ty;

  #define input_image(i2, i1, i0) image[(i2) * (Width) * 3 + (i1) * 3 + (i0)]

  if (image_x_pos < Width && image_y_pos < Height) {
    for(int i = 0; i < 3; i++) {
      x = 255 * (cdf[input_image(image_y_pos, image_x_pos, i)] - cdfmin)/(1.0 - cdfmin);
      input_image(image_y_pos, image_x_pos, i) = min(max(x, start), end);
    }
  }

  #undef input_image
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
  /* declare addtional pointers here */
  float *deviceInputImageData;
  float *deviceOutputImageData;
  unsigned char * device_ucharImage;
  unsigned char * device_grayscaleImage;
  int imageRGBInputSize; 
  int imageGrayscaleInputSize;
  unsigned int * device_histogram;
  float * device_cdf;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile); // brings image in as wbImage_t data type
  /* extracts information about the image */
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels); // allocates memory as wbImage_t data type for final image output
  hostInputImageData = wbImage_getData(inputImage); // allocates and copies data from inputImage from wbImage_t to float * data type
  hostOutputImageData = wbImage_getData(outputImage); // allocates and "gets" data of outputImage into hostOutputImageData but data retrieved is to be overwritten by kernel
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  /* allocate device memory */
  imageRGBInputSize = imageWidth * imageHeight * imageChannels;
  imageGrayscaleInputSize = imageWidth * imageHeight;
  wbCheck(cudaMalloc((void **)&deviceInputImageData, imageRGBInputSize * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, imageRGBInputSize * sizeof(float)));
  wbCheck(cudaMalloc((void **)&device_ucharImage, imageRGBInputSize * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&device_grayscaleImage, imageGrayscaleInputSize * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&device_histogram, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **)&device_cdf, HISTOGRAM_LENGTH * sizeof(float)));

  unsigned char* hostUchar;
  hostUchar = (unsigned char *)malloc(imageRGBInputSize * sizeof(unsigned char));

  /* copy data to device */
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageRGBInputSize * sizeof(float), cudaMemcpyHostToDevice));
  // wbCheck(cudaMemcpy(device_histogram, &host_histogram, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

  /* set up and launch float to unsigned char cast kernel*/
  dim3 dimGrid(ceil((float)imageRGBInputSize / (1.0*IMAGE_BLOCK_SIZE)), 1, 1);
  dim3 dimBlock(IMAGE_BLOCK_SIZE, 1, 1);

  floatToUnsignedChar<<<dimGrid, dimBlock>>>(deviceInputImageData, device_ucharImage, imageRGBInputSize);
  cudaDeviceSynchronize();

  /* create a grayscale image from the unsigned char RGB image */
  dim3 newdimGrid(ceil((float)imageWidth/BLOCK_SIZE), ceil((float)imageHeight/BLOCK_SIZE),1);
  dim3 newdimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  dimGrid = newdimGrid;
  dimBlock = newdimBlock;

  RGBToGrayScale<<<dimGrid, dimBlock>>>(device_ucharImage, device_grayscaleImage, imageHeight, imageWidth);
  cudaDeviceSynchronize();

  /* create a histogram of values from the grayscale image */
  histrogram_privatized_kernel<<<dimGrid, dimBlock>>> (device_grayscaleImage, device_histogram, imageHeight, imageWidth);
  cudaDeviceSynchronize();
  //code to see what the histogram prints 
  unsigned int* temp;
  temp = (unsigned int *)malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemcpy(temp, device_histogram, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  cudaMemcpy(hostUchar, device_ucharImage, imageRGBInputSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  if(imageHeight == 256 && imageWidth == 256){
    printf("\nRGB: %f, %f, %f\n", hostInputImageData[0], hostInputImageData[1], hostInputImageData[2]);
    printf("Uchar: %u, %u, %u\n", hostUchar[0], hostUchar[1], hostUchar[2]);
  }
  printf("This is the histogram\n");
  for(int i = 0; i < HISTOGRAM_LENGTH; i++){
    printf("%d:%u,", i, temp[i]);
  }
  /* turn histogram into probabilities */

//code to see what the histogram prints 
  // printf("This is the histogram\n");
  // for(int i = 0; i < HISTOGRAM_LENGTH; i++){
  //   printf("%d:%u,", i, host_histogram[i]);
  // }

  /* calculate cdf from histogram */
  int numBlocks = ceil((float)HISTOGRAM_LENGTH/(2*BLOCK_SIZE));
  dim3 newdimGrid2(numBlocks, 1, 1);
  dim3 newdimBlock2(BLOCK_SIZE, 1, 1);
  dimGrid = newdimGrid2;
  dimBlock = newdimBlock2;
  scan<<<dimGrid, dimBlock>>>(device_histogram, device_cdf, HISTOGRAM_LENGTH);
  cudaDeviceSynchronize();

  // /* apply color correction to unsigned char of the RGB image */
  dim3 newdimGrid3(ceil((float)imageWidth/BLOCK_SIZE), ceil((float)imageHeight/BLOCK_SIZE));
  dim3 newdimBlock3(BLOCK_SIZE, BLOCK_SIZE, 1);
  dimGrid = newdimGrid3;
  dimBlock = newdimBlock3;

  colorCorrection<<<dimGrid, dimBlock>>>(device_cdf, device_ucharImage, imageHeight, imageWidth);
  cudaDeviceSynchronize();

  // /* cast uchar RGB image back to a float */
  dim3 newdimGrid4(ceil((float)imageRGBInputSize / IMAGE_BLOCK_SIZE), 1, 1);
  dim3 newdimBlock4(IMAGE_BLOCK_SIZE, 1, 1);
  dimGrid = newdimGrid4;
  dimBlock = newdimBlock4;

  unsignedCharToFloat<<<dimGrid, dimBlock>>>(device_ucharImage, deviceOutputImageData, imageRGBInputSize);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageRGBInputSize, cudaMemcpyDeviceToHost);



  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}
