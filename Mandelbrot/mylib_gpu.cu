#include "mylib.h"
#include "mylib_gpu.h"
#include "timing.h"
#include <complex>
#include <iostream>
#include <vector>           // vector
#include <cuComplex.h>
#include <thrust/complex.h>

using namespace std;

__device__ void countIterationsUntilDivergent_GPU(double const &cr, double const &ci, int const threshold, int &iteration)
{
    double zr=0.0;
    double zi=0.0;
    double norm = 0.0;
    iteration = 0;

    while (norm <= 4.0*4.0 && iteration < threshold) {
        //z = (z * z) + c;
        double tmp=zr*zr-zi*zi+cr;
        zi = 2.0*zr*zi+ci;
	zr = tmp;
        norm = zr*zr+zi*zi;        // |z|^2
        ++iteration;
    }
    
    //for (iteration=0; iteration < threshold; ++iteration)
    //{
        ////z = (z * z) + c;
	    //double tmp=zr*zr-zi*zi+cr;
        //zi = 2.0*zr*zi+ci;
	    //zr = tmp;
	    //norm = zr*zr+zi*zi;
	    //if (norm>4.0*4.0) break;
    //}
    //if (iteration < threshold) iteration++;
    
    //printf("  %d  %f  ",iteration,norm);
}

// ---------------------------------------------------------------------------------------
__global__ void mandelbrot_kernel_GPU(int const threshold,
                         double const realAxis[],      int const nreal,
                         double const imaginaryAxis[], int const nimag,
                         int atlas[], int const natlas)
{
// color each point in the atlas depending on the iteration count
    int       idx = blockIdx.x * blockDim.x + threadIdx.x;
    int const str = gridDim.x * blockDim.x;
    
    while (idx<natlas)
    {
        int const ix = idx/nimag;
        int const iy = idx % nimag;
        countIterationsUntilDivergent_GPU(realAxis[ix], imaginaryAxis[iy], threshold, atlas[idx]);
        idx+=str;
    }
}
//---------------------------------------------------------------------------------------
// doens't work with std::complex<T> --> thrust::complex<T>
__device__ void countIterationsUntilDivergent_GPU2(int idx, thrust::complex<double> const c,int const threshold, int &iteration)
{
    thrust::complex<double> z(0.0,0.0);
    iteration = 0;
    while (norm(z) <= 4.0*4.0 && iteration < threshold) {
        z = (z * z) + c;
        ++iteration;
        //if (idx==11) printf("  %d  %f |  ",iteration,znorm);
    }
}

__global__ void mandelbrot_kernel_GPU2(int const threshold,
                         double const realAxis[],      int const nreal,
                         double const imaginaryAxis[], int const nimag,
                         int atlas[], int const natlas)
{
// color each point in the atlas depending on the iteration count
    int       idx = blockIdx.x * blockDim.x + threadIdx.x;
    int const str = gridDim.x * blockDim.x;
    
    while (idx<natlas)
    {
        int const ix = idx/nimag;
        int const iy = idx % nimag;
        thrust::complex<double> c(realAxis[ix], imaginaryAxis[iy]);
        countIterationsUntilDivergent_GPU2(idx, c, threshold, atlas[idx]);
        idx+=str;
    }
}

// ---------------------------------------------------------------------------------------

vector<vector<int>> mandelbrot_split_GPU(int const threshold, int const density)
{
    tic();    
    // location and size of the atlas rectangle
    vector<double> realAxis(density);
    vector<double> imaginaryAxis(density);
    // 2-D --> 1D vector to represent mandelbrot atlas
    vector<int> atlas1d(realAxis.size()*imaginaryAxis.size());
    // 2-D vector to represent mandelbrot atlas
    tic();
    vector<vector<int>> atlas(realAxis.size(), vector<int>(imaginaryAxis.size()));
    cout << "          allocate vector<vector<int>> [sec]: " << toc() << endl;
    
    for (int i = 0; i < density; ++i) {
        realAxis[i] = -0.22 + (i * (0.001 / density));
        imaginaryAxis[i] = -0.70 + (i * (0.001 / density));
    }
    //cout << "realAxis     : " << realAxis << endl;
    //cout << "imaginaryAxis: " << imaginaryAxis << endl;
    int const nreal = static_cast<int>(realAxis.size());
    int const nimag = static_cast<int>(imaginaryAxis.size());

    //int const BLOCK_SIZE=16*16;
    int const BLOCK_SIZE=32;
    //cudaDeviceProp prop;
    //// https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__DEVICE_g5aa4f47938af8276f08074d09b7d520c.html
    //cudaError_t err = cudaGetDeviceProperties (&prop, 0);
    //int const GRID_SIZE=4*prop.multiProcessorCount;
    int const GRID_SIZE=(nreal*nimag + BLOCK_SIZE - 1)/BLOCK_SIZE; 
    
    double *r_d, *i_d;
    int    *a_d;       // device data
    cudaMalloc((void **) &r_d, nreal*sizeof(realAxis[0])); // allocate on device
    cudaMemcpy(r_d, realAxis.data(), nreal*sizeof(realAxis[0]), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &i_d, nimag*sizeof(imaginaryAxis[0]));
    cudaMemcpy(i_d, imaginaryAxis.data(), nreal*sizeof(imaginaryAxis[0]), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &a_d, nreal*nimag*sizeof(atlas1d[0]));
    cout << "      pre kernel [sec]: " << toc() << endl;

    tic();
    // faster (2x)
    mandelbrot_kernel_GPU<<<GRID_SIZE,BLOCK_SIZE>>>(threshold, r_d, nreal, i_d, nimag, a_d, nreal*nimag);
    // slower then previous
    //mandelbrot_kernel_GPU2<<<GRID_SIZE,BLOCK_SIZE>>>(threshold, r_d, nreal, i_d, nimag, a_d, nreal*nimag);
    
    cudaMemcpy(atlas1d.data(),a_d, nreal*nimag*sizeof(atlas1d[0]), cudaMemcpyDeviceToHost);
    // GPU timing
    cudaDeviceSynchronize() ;
    //cout << "atlas1d: " << atlas1d << endl;
    cout << "       in kernel [sec]: " << toc() << endl;
    
    tic();
    cudaFree(a_d);
    cudaFree(i_d);
    cudaFree(r_d);
    // 1D--> 2D
    int idx=0;
    for (int ix = 0; ix < nreal; ++ix) {
        for (int iy = 0; iy < nimag; ++iy) {
            atlas[ix][iy] = atlas1d[idx++];
        }
    }
    cout << "     post kernel [sec]: " << toc() << endl;

    return atlas;
}
// ------------------------------------------------------------------------------------------
// from phind.com
//__global__ void mandelbrotKernel(int width, int height, int max_iterations, double *result) {
__global__ void mandelbrotKernel(
       double const *vcx, int const width,
       double const *vcy, int const height, int max_iterations, int *result) {
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (kx >= width || ky >= height) return;
    
    //double c_x = (kx / (double)width) * 3.5f - 2.5f;
    //double c_y = (ky / (double)height) * 2.0f - 1.0f;
    double c_x = vcx[kx];
    double c_y = vcy[ky];
    //printf("%f : ",c_x);
    
    double z_x = 0.0f;
    double z_y = 0.0f;
    int iter = 0; // GH
    for (iter = 0; iter < max_iterations; iter++) { // GH
    //for (int iter = 0; iter < max_iterations; iter++) {
        double temp = z_x * z_x - z_y * z_y + c_x;
        z_y = 2.0f * z_x * z_y + c_y;
        z_x = temp;
        
        if (z_x * z_x + z_y * z_y > 4.0f*4.0f) break;
    }
    
    result[ky * width + kx] = iter;
}

#define BLOCK_SIZE 16
vector<vector<int>> mandelbrot_split_GPU_phind(int const threshold, int const density)
{
    tic();
    // location and size of the atlas rectangle
    vector<double> realAxis(density);
    vector<double> imaginaryAxis(density);
    // 2-D --> 1D vector to represent mandelbrot atlas
    vector<int> atlas1d(realAxis.size()*imaginaryAxis.size());
    // 2-D vector to represent mandelbrot atlas
    vector<vector<int>> atlas(realAxis.size(), vector<int>(imaginaryAxis.size()));
    
    for (int i = 0; i < density; ++i) {
        realAxis[i] = -0.22 + (i * (0.001 / density));
        imaginaryAxis[i] = -0.70 + (i * (0.001 / density));
    }
    //cout << "realAxis     : " << realAxis << endl;
    //cout << "imaginaryAxis: " << imaginaryAxis << endl;
    int const nreal = static_cast<int>(realAxis.size());
    int const nimag = static_cast<int>(imaginaryAxis.size());
    //cout << density << "   " << nreal << "   " << nimag << endl;

    const int WIDTH = nreal;
    const int HEIGHT = nimag;
    // Kernel launch configuration
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, 
                    (HEIGHT + blockSize.y - 1) / blockSize.y);
    
    
    double *r_d, *i_d;
    int    *a_d;       // device data
    cudaMalloc((void **) &r_d, nreal*sizeof(realAxis[0])); // allocate on device
    cudaMemcpy(r_d, realAxis.data(), nreal*sizeof(realAxis[0]), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &i_d, nimag*sizeof(imaginaryAxis[0]));
    cudaMemcpy(i_d, imaginaryAxis.data(), nreal*sizeof(imaginaryAxis[0]), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &a_d, nreal*nimag*sizeof(atlas1d[0]));
    cout << "      pre kernel [sec]: " << toc() << endl;

    tic();
    mandelbrotKernel<<<gridSize, blockSize>>>(r_d, WIDTH, i_d, HEIGHT, threshold, a_d);
    
    cudaMemcpy(atlas1d.data(),a_d, nreal*nimag*sizeof(atlas1d[0]), cudaMemcpyDeviceToHost);
    // GPU timing
    cudaDeviceSynchronize() ;
    cout << "atlas1d: " << atlas1d << endl;
    cout << "       in kernel [sec]: " << toc() << endl;
    
    tic();
    cudaFree(a_d);
    cudaFree(i_d);
    cudaFree(r_d);
    // 1D--> 2D
    int idx=0;
    for (int ix = 0; ix < nreal; ++ix) {
        for (int iy = 0; iy < nimag; ++iy) {
            atlas[ix][iy] = atlas1d[idx++];
        }
    }
    cout << "     post kernel [sec]: " << toc() << endl;

    return atlas;
}
// 
