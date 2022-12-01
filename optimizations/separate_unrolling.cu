#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define UNROLL_BLOCK_SIZE 256
#define BLOCK_SIZE 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static int max_threads_per_block;
static int max_shared_size;
static int warp_size;

__constant__ float mask_kernel[3136];

__global__ void unroll_Kernel (int C, int H, int W, int K, const float* X, float* X_unroll) {
    // C = number of channels which should always be 3
    // H = height of input images
    // W = width of input images
    // n = current batch sample
    // X = array to a batch's input images, an image for each channel (so 3 images)
    // X_unroll = output where the input images will get mapped to
    // if ((threadIdx.x + threadIdx.y + threadIdx.z == 0) && (blockIdx.x + blockIdx.y + blockIdx.z == 0))
        // printf("Unroll kernel runs");
    int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
    int t = blockIdx.x * UNROLL_BLOCK_SIZE + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;
    int H_unroll = C*K*K;

    int n = blockIdx.y; // the current batch sample

    // #define X_unroll_output(i1, i0) X_unroll[(i1) * (C * K * K) + (i0)]
    #define X_unroll_output(i2, i1, i0) X_unroll[(i2) * (W_out*H_out*C*K*K) + (i1) * (W_out * H_out) + (i0)]
    // #define X_input(i2, i1, i0) X[(i2) * (C * W) + (i1) * W + (i0)]
    #define in_4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    if (t < C * W_unroll) {
        c = t / W_unroll; // channel based on index
        s = t % W_unroll; // section currently working with 
        // c = t / H_unroll; // channel based on index
        // s = t % H_unroll; // section currently working with 
        h_out = s / W_out;
        w_out = s % W_out;
        // h_unroll = h_out * W_out + w_out;
        w_unroll = h_out * W_out + w_out;
        w_base = c * K * K;

        for (p = 0; p < K; p++) {
            for (q = 0; q < K; q++) {
                // w_unroll = w_base + p*K + q;
                h_unroll = w_base + p*K + q;
                // X_output(h_unroll, w_unroll) = X_input(c, h_out + p, w_out + q);
                X_unroll_output(n, h_unroll, w_unroll) = in_4d(n, c, h_out + p, w_out + q);
            }
        }
    }

    #undef X_unroll_output
    #undef in_4d
}

/* code for register tiling of matrix multiplication */
#define TILE_SZ_A 32
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)

__global__ void mygemm(float * __restrict__ c, //<! [out] and MxN matrix
                       const float *a,        //<! [in] an MxK matrix
                       const float *b,        //<! [in] an KxN matrix
                       const int M, const int K, const int N, const int K_in, const int Channel, int H, int W) {

// Macros for accessing flattened matrices
#define A(i1, i0) a[(i1) * K + (i0)] // this will be the mask
#define B(i2, i1, i0) b[(i2) * (K*N) + (i1)*N + (i0)]
#define C(i2, i1, i0) c[(i2)*(M*N) + (i1)*N + (i0)]
// #define A(i1, i0) mask_kernel[(i1) * K + (i0)]

#define in_4d(i3, i2, i1, i0) b[(i3) * (Channel * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

// if ((threadIdx.x + threadIdx.y + threadIdx.z == 0) && (blockIdx.x + blockIdx.y + blockIdx.z == 0))
        // printf("Gemm kernel runs");

int curr_batch = blockIdx.z;

  // Shared memory for tiling input B array
  __shared__ float B_s[TILE_SZ_RATIO][TILE_SZ_B];

  // Index variables
  const unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int col = blockIdx.y * TILE_SZ_B;

  int W_out = W - K_in + 1;
  int H_out = H - K_in + 1;

  // Privatization of output variables
  float c_reg[TILE_SZ_B];

  // Initialize output values
  for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
    c_reg[outIdx] = 0;
  }

  const unsigned int i = threadIdx.x / TILE_SZ_B;
  const unsigned int j = threadIdx.x % TILE_SZ_B;

  // Loop over the input tiles
  for (unsigned int tileIdx = 0; tileIdx < ceil(K/(1.0 * TILE_SZ_RATIO)); ++tileIdx) {
    // Load the tile of B into shared memory
    if (tileIdx * TILE_SZ_RATIO + i < K && col + j < N) {
        B_s[i][j] = B(curr_batch, tileIdx * TILE_SZ_RATIO + i, col + j);
    } else {
        B_s[i][j] = 0;
    }
    __syncthreads();
    // Loop over elements inside the tile
    for (unsigned int idx = 0; idx < TILE_SZ_RATIO; ++idx) {
      // Load tile of A matrix into register
      float a_reg;
      if (row < M && tileIdx * TILE_SZ_RATIO + idx < K) {
        a_reg = A(row, tileIdx * TILE_SZ_RATIO + idx);
      } else {
        a_reg  = 0;
      }
      // Loop over and update the output elemena_regts assigned to the thread
      for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
        c_reg[outIdx] += a_reg * B_s[idx][outIdx];
      }
    }
    __syncthreads();
  }

  for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
    if (row < M && col + outIdx < N) {
      C(curr_batch, row, col + outIdx) = c_reg[outIdx];
    }
  }

#undef A
#undef in_4d
#undef B
#undef C
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    // get_device_properties(); // print and save some of the properties of the gpu
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int inputSize = Batch*Channel*Height*Width*sizeof(float);
    int outputSize = Batch*Map_out*Height_out*Width_out*sizeof(float);
    int maskSize = Map_out*Channel*K*K*sizeof(float);
    cudaMalloc((void **)device_input_ptr, inputSize);
    cudaMalloc((void **)device_mask_ptr, maskSize);
    cudaMalloc((void **)device_output_ptr, outputSize);

    /* optimization that places mask into constant memory */
    int kernel_size = sizeof(float) * Map_out * Channel * K * K;
    // cudaMemcpyToSymbol(mask_kernel, host_mask, kernel_size);

    cudaMemcpy(*device_input_ptr, host_input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, maskSize, cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int W_out = Width - K + 1;
    int H_out = Height - K + 1;
    int H_unroll = Channel * K * K;
    int W_unroll = H_out * W_out;
    // float* X_unrolled = (float*)malloc(W_unroll * H_unroll * sizeof(float));
    float* X_unrolled_device;
    
    gpuErrchk(cudaMalloc((void **)&X_unrolled_device, sizeof(float)*W_unroll*H_unroll*Batch));

    // printf("Global memory is %lu\n", sizeof(float) * W_unroll * H_unroll * Batch);
    // printf("Gemm: gridx = %lu; gridy = %lu; gridz = %lu\n", (unsigned long)(ceil(Map_out / (1.0*TILE_SZ_A))), (unsigned long)(ceil(W_unroll / (1.0*TILE_SZ_B))), (unsigned long)(Batch));
    // printf("Shared memory size is %d", sizeof(float)*TILE_SZ_RATIO*TILE_SZ_B);

    /* now throw it into the matrix multiplication */

    int num_blocks = ceil((float)(H_out*W_out*Channel) / UNROLL_BLOCK_SIZE);
    dim3 dimUGrid(num_blocks, Batch, 1);
    dim3 dimUBlock(UNROLL_BLOCK_SIZE, 1, 1);
    unroll_Kernel<<<dimUGrid, dimUBlock>>>(Channel, Height, Width, K, device_input, X_unrolled_device);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    dim3 dimGrid(ceil(Map_out / (1.0*TILE_SZ_A)), ceil(W_unroll / (1.0*TILE_SZ_B)), Batch);
    dim3 dimBlock(TILE_SZ_A, 1, 1);
    mygemm<<<dimGrid, dimBlock>>>(device_output, device_mask, X_unrolled_device, Map_out, Channel*K*K, H_out*W_out, K, Channel, Height, Width);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    cudaDeviceSynchronize();

    cudaFree(X_unrolled_device);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int outputSize = Batch*Map_out*Height_out*Width_out*sizeof(float);

    // Copy the output back to host
    cudaMemcpy(host_output, device_output, outputSize, cudaMemcpyDeviceToHost);

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
        max_shared_size = deviceProp.sharedMemPerBlock;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        max_threads_per_block = deviceProp.maxThreadsPerBlock;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
        warp_size = deviceProp.warpSize;
    }
}
