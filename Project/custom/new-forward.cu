#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define UNROLL_BLOCK_SIZE 256
#define BLOCK_SIZE 1024

static int max_threads_per_block;
static int max_shared_size;
static int warp_size;

__constant__ float mask_kernel[3136];

/* original simple convolution kernel */
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
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
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

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
    n = blockIdx.x; // current batch sample
    m = blockIdx.y; // current map output feature
    h0 = threadIdx.y; 
    w0 = threadIdx.x;
    h_base = (blockIdx.z/W_grid)*tile_width; // vertical base out data index for the block
    w_base = (blockIdx.z % W_grid)*tile_width; // horizontal base out data index for the block 
    h = h_base + h0;
    w = w_base + w0;

    float acc = 0; // initialize output result

    /* iterate through all the channels */
    for (int c = 0; c < Channel; c++) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                // if (h_base + h0 + p < Height && w_base + w0 + q < Width)
                    // acc += in_4d(n, c, h_base + h0 + p, w_base + w0 + q) * mask_4d(m, c, p, q);
                    acc += in_4d(n, c, h_base + h0 + p, w_base + w0 + q) * const_mask_4d(m, c, p, q);
            }
        }
    }
    if ((h < Height_out) && (w < Width_out)) {
        out_4d(n, m, h, w) = acc;
    }
    
    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef const_mask_4d
}

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
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

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

    int n = blockIdx.y; // the current batch sample

    // #define X_unroll_output(i1, i0) X_unroll[(i1) * (C * K * K) + (i0)]
    #define X_unroll_output(i2, i1, i0) X_unroll[(i2) * (W_out*H_out*C*K*K) + (i1) * (W_out * H_out) + (i0)]
    // #define X_input(i2, i1, i0) X[(i2) * (C * W) + (i1) * W + (i0)]
    #define in_4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    if (t < C * W_unroll) {
        c = t / W_unroll; // channel based on index
        s = t % W_unroll; // section currently working with 
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

    // if (t < W_unroll) {
    //     // c = t / W_unroll; // channel based on index
    //     // s = t % W_unroll; // section currently working with 
    //     // h_out = s / W_out;
    //     // w_out = s % W_out;
    //     // // h_unroll = h_out * W_out + w_out;
    //     // w_unroll = h_out * W_out + w_out;
    //     // w_base = c * K * K;

    //     // for (p = 0; p < K; p++) {
    //     //     for (q = 0; q < K; q++) {
    //     //         // w_unroll = w_base + p*K + q;
    //     //         h_unroll = w_base + p*K + q;
    //     //         // X_output(h_unroll, w_unroll) = X_input(c, h_out + p, w_out + q);
    //     //         X_unroll_output(h_unroll, w_unroll) = in_4d(n, c, h_out + p, w_out + q);
    //     //     }
    //     // }

    // }

    #undef X_unroll_output
    #undef in_4d
}

__host__ void unroll_gpu(int C, int H, int W, int K, int n, const float* X, float* X_unroll) 
{
    int H_out = H - K + 1;
    int W_out = W - K + 1; 
    //  num_threads = C * H_out * W_out; // number of total threads used
     int num_blocks = ceil((float)(C * H_out * W_out) / UNROLL_BLOCK_SIZE);
    // int num_blocks = ceil((float)(H_out * W_out) / UNROLL_BLOCK_SIZE);
    dim3 dimGrid(num_blocks, n, 1);
    dim3 dimBlock(UNROLL_BLOCK_SIZE, 1, 1);
    unroll_Kernel<<<dimGrid, dimBlock>>>(C, H, W, K, X, X_unroll);
}

/* code for register tiling of matrix multiplication */
#define TILE_SZ_A 64
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)

__global__ void mygemm(float * __restrict__ c, //<! [out] and MxN matrix
                       const float *a,        //<! [in] an MxK matrix
                       const float *b,        //<! [in] an KxN matrix
                       const int M, const int K, const int N, const int K_in, const int Channel, int H, int W) {

// Macros for accessing flattened matrices
#define A(i1, i0) a[ (i1) * K + (i0)] // this will be the mask
#define B(i2, i1, i0) b[(i2) * (K*N) + (i1)*N + (i0)]
#define C(i2, i1, i0) c[(i2)*(M*N) + (i1)*N + (i0)]

#define in_4d(i3, i2, i1, i0) b[(i3) * (Channel * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

// if ((threadIdx.x + threadIdx.y + threadIdx.z == 0) && (blockIdx.x + blockIdx.y + blockIdx.z == 0))
        printf("Gemm kernel runs");

int curr_batch = blockIdx.z;

  // Shared memory for tiling input B array
  __shared__ float B_s[TILE_SZ_RATIO][TILE_SZ_B];

  // Index variables
  const unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int col = blockIdx.y * TILE_SZ_B;

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
        // int curr_channel = (tileIdx * TILE_SZ_RATIO + i) / Channel;
        // int m = (col + j) % K_in;
        // int n = (col + j) / K_in;
        // int p = ((tileIdx * TILE_SZ_RATIO + i) - (curr_channel*K_in*K_in)) % K_in;
        // int q = ((tileIdx * TILE_SZ_RATIO + i) - (curr_channel*K_in*K_in)) / K_in;
        // int x = m + p;
        // int y = n + q;
        B_s[i][j] = B(curr_batch, tileIdx * TILE_SZ_RATIO + i, col + j);
        // B_s[i][j] = B(curr_batch, y, x);
        // B_s[i][j] = in_4d(curr_batch, curr_channel, y, x);
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

/* convolution with shared memory and matrix multiplication */
__host__ void unroll_multiply(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int W_out = Width - K + 1;
    int H_out = Height - K + 1;
    int H_unroll = Channel * K * K;
    int W_unroll = H_out * W_out;
    // float* X_unrolled = (float*)malloc(W_unroll * H_unroll * sizeof(float));
    float* X_unrolled_device;
    // float* host_input = (float*)malloc(Batch*Channel*Height*Width*sizeof(float));
    // float* host_mask = (float*)malloc(Map_out*Channel*K*K*sizeof(float));
    // float* host_output = (float*)malloc(Batch*Map_out*H_out*W_out*sizeof(float));

    // cudaMemcpy(host_input, input, Batch*Channel*Height*Width*sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(host_mask, mask, Map_out*Channel*K*K*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMalloc((void **)&X_unrolled_device, sizeof(float)*W_unroll*H_unroll*Batch);

    #define out_4d(i3, i2, i1, i0) host_output[(i3) * (Map_out * W_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define out_unrolled(i2, i1, i0) host_output[(i2) * (Map_out * H_out * W_out) + (i1) * (H_out * W_out) + (i0)] // i2 = batch number, 
    #define in_4d(i3, i2, i1, i0) host_input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) host_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define mask_unrolled(i1, i0) host_mask [(i1) * W_unroll + (i0)] // i1 = Map output feature, i0 = mask value
    #define X_unroll_output(i1, i0) X_unrolled[(i1) * W_unroll + (i0)] // i1 = one unrolled input for one output map feature

    int numARows = Map_out;
    int numACols = Channel*K*K;
    int numBRows = H_unroll;
    int numBCols = W_unroll;
    int numCRows = Map_out;
    int numCCols = H_out*W_out;

    dim3 dimGrid(ceil(Map_out / (1.0*TILE_SZ_A)), ceil(W_unroll / (1.0*TILE_SZ_B)), Batch);
    dim3 dimBlock(TILE_SZ_A, 1, 1);

    printf("Global memory is %lu\n", sizeof(float) * W_unroll * H_unroll * Batch);
    printf("Gemm: gridx = %lu; gridy = %lu; gridz = %lu\n", (unsigned long)(ceil(Map_out / (1.0*TILE_SZ_A))), (unsigned long)(ceil(W_unroll / (1.0*TILE_SZ_B))), (unsigned long)(Batch));
    printf("Shared memory size is %d", sizeof(float)*TILE_SZ_RATIO*TILE_SZ_B);

    // for (int n = 0; n < Batch; n++) {
        unroll_gpu(Channel, Height, Width, K, Batch, input, X_unrolled_device);

        /* now throw it into the matrix multiplication */
        mygemm<<<dimGrid, dimBlock>>>(output, mask, X_unrolled_device, Map_out, Channel*K*K, H_out*W_out, K, Channel, Height, Width);
        printf("\nBegin:\n");
        mygemm<<<dimGrid, dimBlock>>>(output, mask, input, Map_out, Channel*K*K, H_out*W_out, K, Channel, Height, Width);
        printf("\nEnd\n");
        cudaDeviceSynchronize();
        // cudaMemcpy(X_unrolled, X_unrolled_device, sizeof(float)*W_unroll*H_unroll, cudaMemcpyDeviceToHost);

        // /* basic cpu matrix multiply to check correctness of unrolling of input */
        // // MatrixMultipy(H_unroll, W_unroll, X_unrolled, n, m, Channel, Map_out, K, Width, mask, output);
        // for (int m_mask = 0; m_mask < Map_out; m_mask++) {
        //     for (int m_x = 0; m_x < H_unroll; m_x++) {
        //         int acc = 0;
        //         for (int i = 0; i < W_unroll; i++) {
        //             acc += mask_unrolled(m_mask, i) * X_unroll_output(m_x, i);
        //         }
        //         out_unrolled(n, m_mask, m_x) = acc;
        //     }
        // }
    // }

    // cudaMemcpy(output, host_output, Batch*Map_out*H_out*W_out*sizeof(float), cudaMemcpyHostToDevice);

    cudaFree(X_unrolled_device);

    #undef out_4d
    #undef out_unrolled
    #undef in_4d
    #undef mask_4d
    #undef mask_unrolled
    #undef X_unroll_output
}

__global__ void unroll_kernel_fusion(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {

    int H_out = Height - K + 1;
    int W_out = Width - K + 1;

    int tx = threadIdx.x; // current output being calculated between 0-BLOCK_SIZE-1
    int n = blockIdx.x; // current batch sample
    int by = blockIdx.y; // current map output feature
    int offset = blockIdx.z * BLOCK_SIZE; // the offset of each thread

    extern __shared__ float shmem[];
    float* input_tile = &shmem[0];
    /* original */
    // float* smask = &shmem[BLOCK_SIZE * K * K];
    /* with max threads */
    float* smask = &shmem[BLOCK_SIZE * K];

    #define in_tile(i1, i0) input_tile[(i1) * BLOCK_SIZE + (i0)]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define const_mask_4d(i3, i2, i1, i0) mask_kernel[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * W_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

    /* iterate through all of the channels*/
    
    // for (int offset = 0; offset + tx < H_out * W_out; offset += BLOCK_SIZE) {
        int curr_w_out = (offset + tx) % W_out;
        int curr_h_out = (offset + tx) / H_out;
        float acc = 0;
        for (int c = 0; c < Channel; c++) {
            for (int k_i = 0; k_i < K; k_i++) {

                /* load input tile and mask into shared memory */

                /* original */
                // for (int i = 0; i < K; i++) {
                //     for (int j = 0; j < K; j++) {
                //         int w = curr_w_out + i;
                //         int h = curr_h_out + j;
                //         if (h < Height && w < Width)
                //             in_tile(j*K + i, tx) = in_4d(n, c, h, w);
                //         else
                //             in_tile(j*K + i, tx) = 0;
                //     }
                // }

                /* max threads */
                for (int i = 0; i < K; i++) {
                    int w = curr_w_out + i;
                    int h = curr_h_out + k_i;
                    if (h < Height && w < Width)
                        in_tile(i, tx) = in_4d(n, c, h, w);
                    else
                        in_tile(i, tx) = 0;
                }
                
                

                /* load mask into shared memory */

                /* original */
                // if (tx < K*K) {
                //     // int i = tx % K;
                //     // int j = tx / K;
                //     // smask[tx] = mask_4d(by, c, j, i);
                //     smask[tx] = mask[(by) * (Channel * K * K) + (c) * (K * K) + tx];
                // }

                /* with max threads */
                if (tx < K) {
                    // smask[tx] = mask_4d(by, c, k_i, tx);
                    smask[tx] = const_mask_4d(by, c, k_i, tx);
                }

                __syncthreads();

                /* original */
                // for (int i = 0; i < K*K; i++) {
                //     acc += smask[i] * in_tile(i, tx);
                // }

                /* with max threads */
                for (int i = 0; i < K; i++) {
                    acc += smask[i] * in_tile(i, tx);
                }

                __syncthreads();
            }
        }

        if (curr_h_out < H_out && curr_w_out < W_out) {
            // atomicAdd(&out_4d(n, by, curr_h_out, curr_w_out) , acc);
            out_4d(n, by, curr_h_out, curr_w_out) = acc;
        }
    // }

    #undef in_tile
    #undef mask_4d
    #undef const_mask_4d
    #undef in_4d
    #undef out_4d

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
    // int kernel_size = sizeof(float) * Map_out * Channel * K * K;
    // cudaMemcpyToSymbol(mask_kernel, host_mask, kernel_size);

    // cudaMemcpy(*device_input_ptr, host_input, inputSize, cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, maskSize, cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    get_device_properties();

    /* this is for the original convolution kernel */

    /* this is to find the optimal tile width based on the limitations of the kernel */
    // int optimal_tile_width = sqrt(max_threads_per_block);
    // printf("The optimal tile width is %d\n", optimal_tile_width);

    // // Set the kernel dimensions and call the kernel
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // int N = Batch; // blockDim.x will corresponds to the batch sample
    // int M = Map_out; // blockDim.y corresponds to the output feature
    // // int W_grid = ceil((float)Width_out / TILE_WIDTH);
    // // int H_grid = ceil((float)Height_out / TILE_WIDTH);
    // int W_grid = ceil((float)Width_out / optimal_tile_width);
    // int H_grid = ceil((float)Height_out / optimal_tile_width);
    // int Z = W_grid * H_grid; // the current output tile to be computed in the output feature

    // // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 blockDim(optimal_tile_width, optimal_tile_width, 1);
    // dim3 gridDim(N, M, Z);

    // conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    // cudaDeviceSynchronize();

    /* this is for the convolution kernel with shared memory*/
    /* this is for finding the optimal tile size */
    // int optimal_tile_width = sqrt(max_threads_per_block);
    // int num_blocks = -1;
    // while((optimal_tile_width * optimal_tile_width) > warp_size) {
    //     int shared_mem_size = sizeof(float) * ((optimal_tile_width + K - 1) * (optimal_tile_width + K - 1) + K*K);
        
    //     if (shared_mem_size > max_shared_size) {
    //         optimal_tile_width /= 2;
    //         continue;
    //     }

    //     else
    //         break;
    // }
    // optimal_tile_width = 8;
    // printf("Optimal tile width is %d", optimal_tile_width);
    // // Set the kernel dimensions and call the kernel
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // int N = Batch; // blockDim.x will corresponds to the batch sample
    // int M = Map_out; // blockDim.y corresponds to the output feature
    // int W_grid = ceil((float)Width_out / optimal_tile_width);
    // int H_grid = ceil((float)Height_out / optimal_tile_width);
    // int Z = W_grid * H_grid; // the current output tile to be computed in the output feature

    // dim3 blockDim(optimal_tile_width, optimal_tile_width, 1);
    // dim3 gridDim(N, M, Z);

    // size_t shmem_size = sizeof(float) * ((optimal_tile_width + K - 1) * (optimal_tile_width + K - 1) + K*K);
    // conv_forward_kernel_shared_mem<<<gridDim, blockDim, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    // cudaDeviceSynchronize();

    /* this is shared memory and multiplication */
    // Set the kernel dimensions and call the kernel
    unroll_multiply(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);

    /* this is the kernel fusion for unrolling and doing the matrix multiplication with shared memory */
    // const int H_out = Height - K + 1;
    // const int W_out = Width - K + 1;
    // const int threadGroups = ceil((float)(W_out * H_out) / BLOCK_SIZE);
    // dim3 gridDim(Batch, Map_out, threadGroups);
    // // dim3 blockDim(H_out * W_out, 1, 1);
    // dim3 blockDim(BLOCK_SIZE, 1, 1);
    // /* original */
    // // size_t shmem_size = sizeof(float) * ((BLOCK_SIZE * K * K) + (K * K));
    // /* with max threads */
    // size_t shmem_size = sizeof(float) * ((BLOCK_SIZE * K) + K);
    // printf("Number of threads are %d\n", BLOCK_SIZE);
    // printf("Numbers of blocks are %d, %d\n", Batch, Map_out);
    // printf("K is %d, K*K is %d\n",K, K*K);
    // printf("Max size of mask is: %d\n", K*K*Channel*Map_out*sizeof(float));
    // unroll_kernel_fusion<<<gridDim, blockDim, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    // cudaDeviceSynchronize();

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int outputSize = Batch*Map_out*Height_out*Width_out*sizeof(float);

    // Copy the output back to host
    cudaMemcpy(host_output, device_output, outputSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if(Channel == 1){
        FILE *fp;
        fp = fopen("out.txt", "w");

        for (int i = 0; i < Height_out; i++) {
            for (int j = 0; j < Width_out; j++) {
                fprintf(fp, "%.2f ", host_output[i * Width_out + j]);
            // check for error here too
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
    else {
        FILE *fp;
        fp = fopen("out2.txt", "w");

        for (int i = 0; i < Height_out; i++) {
            for (int j = 0; j < Width_out; j++) {
                fprintf(fp, "%.2f ", host_output[i * Width_out + j]);
            // check for error here too
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
    
    // /* print some text */
    // const char *text = "Write this to the file";
    // fprintf(f, "Some text: %s\n", text);

    // /* print integers and floats */
    // int i = 1;
    // float pi= 3.1415927;
    // fprintf(f, "Integer: %d, float: %f\n", i, pi);

    // /* printing single characters */
    // char c = 'A';
    // fprintf(f, "A character: %c\n", c);

    // fclose(f);

    

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
