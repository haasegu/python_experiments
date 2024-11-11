#pragma once
#include <complex>
#include <vector>

/** Mandelbrot iteration in a certain complex domain using the kernel function on GPU.
 *
 *  @param[in]  threshold      maximal number of iterations
 *  @param[in]  density        number of pixels in x- and y-direction
 *  @return 2d-vector[@p nreal][@p nimag]  with stopping iteration index for each pixel.
 *  @see mandelbrot
*/
std::vector<std::vector<int>> mandelbrot_split_GPU(int const threshold, int const density);

std::vector<std::vector<int>> mandelbrot_split_GPU_phind(int const threshold, int const density);
