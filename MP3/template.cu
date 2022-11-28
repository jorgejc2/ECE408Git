
#include <wb.h>

/* personal defines */
#define TILE_WIDTH 16

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
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  /* used in my tile multiplication */
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  /* identify the column and row of the current thread */
  int row = by * TILE_WIDTH + ty;  //position in the full matrix
  int col = bx * TILE_WIDTH + tx;  //position in the full matrix

  /*
  Remember:
  numAColumns = A's width
  numBColumns = B's width
  numAColumns == numBRows
  */

  float Pvalue = 0;

  /* the number of tiles to loop over is equal to aRows or bColumns */
  /* loop over the M and N tiles required to compute P element */
  for (int ph = 0; ph < ceil(1.0*numAColumns/TILE_WIDTH); ++ph) {

    /* collaborative loading of M and N tiles into shared memory */
    if ((row < numARows) && ((ph*TILE_WIDTH + tx) < numAColumns))
      Mds[ty][tx] = A[row*numAColumns + (ph*TILE_WIDTH + tx)];
    else
      Mds[ty][tx] = 0;

    if (((ph*TILE_WIDTH + ty) < numBRows) && (col < numBColumns))
      Nds[ty][tx] = B[(ph*TILE_WIDTH + ty)*numBColumns + col];
    else
      Nds[ty][tx] = 0;
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++)
      Pvalue += Mds[ty][k] * Nds[k][tx];
    __syncthreads();

    
  }

  if ((row < numCRows) && (col < numCColumns))
      C[row*numCColumns + col] = Pvalue;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  wbTime_stop(Generic, "Importing data and creating memory on host");

  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here

  int aSize = numARows * numAColumns * sizeof(float);
  int bSize = numBRows * numBColumns * sizeof(float);
  int cSize = numCRows * numCColumns * sizeof(float);
  cudaMalloc((void **)&deviceA, aSize);
  cudaMalloc((void **)&deviceB, bSize);
  cudaMalloc((void **)&deviceC, cSize);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  cudaMemcpy(deviceA, hostA, aSize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, bSize, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  dim3 dimGrid(ceil((float)numCColumns/TILE_WIDTH), ceil((float)numCRows/TILE_WIDTH), 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns); 

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostC, deviceC, cSize, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  cudaFree(deviceA);
  cudaFree(deviceB);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
