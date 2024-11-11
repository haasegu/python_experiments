// /usr/local/cuda/bin/nvcc  -g -pg -Xcompiler -fmax-errors=1 -Xcompiler -O3 -std=c++20 --expt-relaxed-constexpr -O3 -use_fast_math -restrict --ftemplate-backtrace-limit 1 -gencode arch=compute_75,code=\"compute_75,sm_75\" -gencode arch=compute_80,code=\"compute_80,sm_80\"  --ptxas-options=-v,-warn-spills --resource-usage -src-in-ptx --restrict --Wreorder --ftemplate-backtrace-limit 1 -res-usage -Wno-deprecated-declarations  --compiler-options=-fopenmp,-O3,-funsafe-math-optimizations  phind_mandelbrot.cu

#include "timing.h"    // GH
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 16
#define MAX_ITERATIONS 1000

__global__ void mandelbrotKernel(int width, int height, int max_iterations, float *result) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float c_x = (x / (float)width) * 3.5f - 2.5f;
    float c_y = (y / (float)height) * 2.0f - 1.0f;
    
    float z_x = 0.0f;
    float z_y = 0.0f;
    int iter = 0; // GH
    for (iter = 0; iter < max_iterations; iter++) { // GH
    //for (int iter = 0; iter < max_iterations; iter++) {
        float temp = z_x * z_x - z_y * z_y + c_x;
        z_y = 2.0f * z_x * z_y + c_y;
        z_x = temp;
        
        if (z_x * z_x + z_y * z_y > 4.0f*4.0f) break;
    }
    
    result[y * width + x] = iter;
}

int main() {
    const int WIDTH = 1920;
    const int HEIGHT = 1080;
    
    // Host variables
    float *h_result;
    cudaMallocHost(&h_result, WIDTH * HEIGHT * sizeof(float));
    
    // Device variables
    float *d_result;
    cudaMalloc(&d_result, WIDTH * HEIGHT * sizeof(float));
    
    // Kernel launch configuration
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, 
                    (HEIGHT + blockSize.y - 1) / blockSize.y);
    
    tic();  // GH
    // Launch kernel
    mandelbrotKernel<<<gridSize, blockSize>>>(WIDTH, HEIGHT, MAX_ITERATIONS, d_result);
    
    // Copy result from device to host
    cudaMemcpy(h_result, d_result, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    double tdiff = toc();  // Gh
    
    // Print result
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("Result[%d,%d] = %d\n", x, y, (int)h_result[y * WIDTH + x]);
        }
    }
    
    // Free device memory
    cudaFree(d_result);
    
    // Free host memory
    cudaFreeHost(h_result);
    
    printf("Run time: %f\n",tdiff);  // GH
    
    return 0;
}
