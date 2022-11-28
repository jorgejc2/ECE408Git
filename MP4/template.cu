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
#define TILE_WIDTH 8
#define MASK_WIDTH 3
#define RADIUS 8 + 3 - 1

//@@ Define constant memory for device kernel here
__constant__ float mask_kernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  /* recall that the matrix is z-major order, so z increments the slowest, then y, the x */

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int col_o = blockIdx.x * TILE_WIDTH + tx; // x-dim output
  int row_o = blockIdx.y * TILE_WIDTH + ty; // y-dim output
  int depth_o = blockIdx.z * TILE_WIDTH + tz; // z-dim output

  int col_i = col_o - (MASK_WIDTH / 2); // x-dim input
  int row_i = row_o  - (MASK_WIDTH / 2); // y-dim input
  int depth_i = depth_o - (MASK_WIDTH / 2); // z-dim input

  int i, j, k; // used to index in the for loops

  __shared__ float Nds[RADIUS][RADIUS][RADIUS];

  float Pvalue = 0;
  /* load shared table with dependent inputs */
  if ( col_i >= 0 && col_i < x_size && row_i >= 0 && row_i < y_size && depth_i >= 0 && depth_i < z_size ) {
    Nds[tz][ty][tx] = input[ (depth_i*x_size*y_size) + (row_i*x_size) + col_i ];
  }
  else {
    Nds[tz][ty][tx] = 0;
  }

  __syncthreads(); // need tile to fully load inputs

  if ((tx < TILE_WIDTH) && (ty < TILE_WIDTH) && (tz < TILE_WIDTH)) {
    for (i = 0; i < MASK_WIDTH; i++) {
      for (j = 0; j < MASK_WIDTH; j++) {
        for (k = 0; k < MASK_WIDTH; k++) {
          Pvalue += mask_kernel[i][j][k] * Nds[i + tz][j + ty][k + tx]; // perform convolution
        }
      }
    }
    /* need to check that current thread is supposed to calculate an output and where it wants to output is in the matrix bounds */
    if ((col_o < x_size) && (row_o < y_size) && (depth_o < z_size))
      output[ (depth_o*x_size*y_size) + (row_o*x_size) + col_o ] = Pvalue;
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
  float *deviceInput;
  float *deviceOutput;

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

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int total_size = x_size*y_size*z_size*sizeof(float);
  int kernel_size = kernelLength*sizeof(float);

  cudaMalloc((void **)&deviceInput, total_size);
  cudaMalloc((void **)&deviceOutput, total_size);

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu

  cudaMemcpyToSymbol(mask_kernel, hostKernel, kernel_size);
  cudaMemcpy(deviceInput, hostInput + 3, total_size, cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid (ceil((float)x_size/TILE_WIDTH), ceil((float)y_size/TILE_WIDTH), ceil((float)z_size/TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);

  //@@ Launch the GPU kernel here
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutput + 3, deviceOutput, total_size, cudaMemcpyDeviceToHost);

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
