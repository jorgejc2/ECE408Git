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

/* convolution with shared memory */
__global__ void conv_forward_kernel_shared_mem(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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

    int tile_width = blockDim.x;

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_grid = ceil((float)Width_out / tile_width);
    const int H_grid = ceil((float)Height_out / tile_width);

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define const_mask_4d(i3, i2, i1, i0) mask_kernel[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    int n, m , h0, w0, h_base, w_base, h, w;
    int X_tile_width = tile_width + K - 1;
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[X_tile_width * X_tile_width];
    n = blockIdx.x; // current batch sample
    m = blockIdx.y; // current map output feature
    h0 = threadIdx.y; 
    w0 = threadIdx.x;
    h_base = (blockIdx.z/W_grid)*tile_width; // vertical base out data index for the block
    w_base = (blockIdx.z % W_grid)*tile_width; // horizontal base out data index for the block 
    h = h_base + h0;
    w = w_base + w0;

    #define x_share(i1, i0) X_shared[(i1)*(X_tile_width) + (i0)]
    #define w_share(i1, i0) W_shared[(i1)*(K) + (i0)]

    float acc = 0; // initialize output result

    /* iterate through all the channels */
    for (int c = 0; c < Channel; c++) {
        if ((h0 < K) && (w0 < K)) {
            // w_share(h0, w0) = mask_4d(m, c, h0, w0);
            w_share(h0, w0) = const_mask_4d(m, c, h0, w0);
        }
        // else
        //     W_shared[h0, w0] = 0;
        // __syncthreads();

        for (int i = h; i < h_base + X_tile_width; i += tile_width) {
            for (int j = w; j < w_base + X_tile_width; j += tile_width) {
                if ((i < Height) && (j < Width)) {
                    x_share(i - h_base, j - w_base) = in_4d(n, c, i, j);
                }
                else {
                    x_share(i - h_base, j - w_base) = 0;
                }
            }
        }
        __syncthreads();

        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                acc += x_share(h0 + p, w0 + q) * w_share(p, q);
            }
        }
        __syncthreads();
    }
    if ((h < Height_out) && (w < Width_out)) {
        out_4d(n, m, h, w) = acc;
    }
    
    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef x_share
    #undef w_share
    #undef const_mask_4d
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
    // cudaMalloc((void **)device_mask_ptr, maskSize);
    cudaMalloc((void **)device_output_ptr, outputSize);

    /* optimization that places mask into constant memory */
    int kernel_size = sizeof(float) * Map_out * Channel * K * K;
    cudaMemcpyToSymbol(mask_kernel, host_mask, kernel_size);

    cudaMemcpy(*device_input_ptr, host_input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, maskSize, cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    get_device_properties(); // DO NOT COMMENT OUT

    int optimal_tile_width = sqrt(max_threads_per_block);
    int num_blocks = -1;
    while((optimal_tile_width * optimal_tile_width) > warp_size) {
        int shared_mem_size = sizeof(float) * ((optimal_tile_width + K - 1) * (optimal_tile_width + K - 1) + K*K);
        
        if (shared_mem_size > max_shared_size) {
            optimal_tile_width /= 2;
            continue;
        }

        else
            break;
    }
    optimal_tile_width = 16;
    printf("Optimal tile width is %d", optimal_tile_width);
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int N = Batch; // blockDim.x will corresponds to the batch sample
    int M = Map_out; // blockDim.y corresponds to the output feature
    int W_grid = ceil((float)Width_out / optimal_tile_width);
    int H_grid = ceil((float)Height_out / optimal_tile_width);
    int Z = W_grid * H_grid; // the current output tile to be computed in the output feature

    dim3 blockDim(optimal_tile_width, optimal_tile_width, 1);
    dim3 gridDim(N, M, Z);

    size_t shmem_size = sizeof(float) * ((optimal_tile_width + K - 1) * (optimal_tile_width + K - 1) + K*K);
    conv_forward_kernel_shared_mem<<<gridDim, blockDim, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
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
    // cudaFree(device_mask);

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
