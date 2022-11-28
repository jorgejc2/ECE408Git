// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

/*
  inputs:
          data -- output from first scan to add values to
          aux_data -- list of values to add for each block in data
          data_len -- number of elements in data
  outputs:
          data -- will be modified to hold final result
  returns:
          none
  effects: 
          Possibly inefficient since first block does no work but simplest implementation
          I could do in five minutes
*/
__global__ void simpleAdd(float *data, float *aux_data, int data_len) {

  __shared__ float T[2*BLOCK_SIZE];
  
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int aux_val = aux_data[bx];

  int input_idx = tx + (2*BLOCK_SIZE * (bx+1));

  /* load shared memory */
  T[tx] = input_idx < data_len ? data[input_idx] : 0;
  T[tx + blockDim.x] = input_idx + blockDim.x < data_len ? data[input_idx + blockDim.x] : 0;

  /* add aux value to every element in the block */
  __syncthreads();
  T[tx] += aux_val;
  T[tx + blockDim.x] += aux_val;

  /* store values back into data for appropriate block */
  __syncthreads();
  if (input_idx < data_len)
    data[input_idx] = T[tx];
  if (input_idx + blockDim.x < data_len)
    data[input_idx + blockDim.x] = T[tx + blockDim.x];
  
}

__global__ void scan(float *input, float *output, int len) {
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

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  /* additional necessary arrays for 2nd part computation */
  float *host_aux_inputs;
  float *device_aux_inputs;
  float *device_aux_outputs;
  int aux_size = ceil((float)numElements/(2*BLOCK_SIZE)) - 1;
  host_aux_inputs = (float *)malloc(aux_size * sizeof(float));

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&device_aux_inputs, aux_size * sizeof(float)));
  wbCheck(cudaMalloc((void **)&device_aux_outputs, aux_size * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int numBlocks = ceil((float)numElements/(2*BLOCK_SIZE));
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements);
  cudaDeviceSynchronize();

  /* need data back onto host to find last value of every block except the last one*/
  cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost);

  /* copy values into aux array */
  for (int i = 0; i < numBlocks - 1; i++) {
    host_aux_inputs[i] = hostOutput[(i+1)*2*BLOCK_SIZE - 1];
  }

  /* must do auxillary step if more than one block was used in the original scan */
  if (numBlocks > 1) {
    /* do scan to calculate auxillary values */
    cudaMemcpy(device_aux_inputs, host_aux_inputs, aux_size * sizeof(float), cudaMemcpyHostToDevice);
    scan<<<1, BLOCK_SIZE>>>(device_aux_inputs, device_aux_outputs, aux_size);
    cudaDeviceSynchronize();

    /* perform a simple add with first scan's output and calculated auxillary values */
    simpleAdd<<<dimGrid, dimBlock>>>(deviceOutput, device_aux_outputs, numElements);
    cudaDeviceSynchronize();
  }
  

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(device_aux_inputs);
  cudaFree(device_aux_outputs);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);
  free(host_aux_inputs);

  return 0;
}
