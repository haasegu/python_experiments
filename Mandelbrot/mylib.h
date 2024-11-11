#pragma once
#include <complex>
#include <iomanip>      // std::setprecision
#include <iostream>
#include <vector>

/** Mandelbrot iteration on one pixel.
 *
 *  @param[in]  c              pixel
 *  @param[in]  threshold      maximal number of iterations
 *  @return number of iterations until loop finishes 
*/
int countIterationsUntilDivergent(std::complex<double> const &c, int const threshold);


/** Mandelbrot iteration in a certain complex domain.
 *
 *  @param[in]  threshold      maximal number of iterations
 *  @param[in]  density        number of pixels in x- and y-direction
 *  @return 2d-vector[@p nreal][@p nimag]  with stopping iteration index for each pixel.
*/
std::vector<std::vector<int>> mandelbrot(int const threshold, int const density);


/** The Mandelbrot iteration in a given rectangular domain.
 *
 *  @param[in]  threshold      maximal number of iterations
 *  @param[in]  realAxis       vector of values on real axis
 *  @param[in]  imaginaryAxis  vector of values on imaginary axis
 *  @param[out] atlas          resulting 2d-vector[@p nreal][@p nimag] with iteration number per pixel
*/
void mandelbrot_kernel(int const threshold, std::vector<double> const &realAxis, std::vector<double> const &imaginaryAxis,
                       std::vector<std::vector<int>>  &atlas);

/** Mandelbrot iteration in a certain complex domain using the kernel function.
 *
 *  @param[in]  threshold      maximal number of iterations
 *  @param[in]  density        number of pixels in x- and y-direction
 *  @return 2d-vector[@p nreal][@p nimag]  with stopping iteration index for each pixel.
 *  @see mandelbrot
*/
std::vector<std::vector<int>> mandelbrot_split(int const threshold, int const density);


/** The Mandelbrot iteration in a given rectangular domain. C-like interface.
 *
 *  @param[in]  threshold      maximal number of iterations
 *  @param[in]  realAxis       array of values on real axis
 *  @param[in]  nreal          number of elements in @p realAxis
 *  @param[in]  imaginaryAxis  array of values on imaginary axis
 *  @param[in]  niamg          number of elements in @p imaginaryAxis
 *  @param[out] atlas          resulting array[@p nreal * @p nimag] with iteration number per pixel
*/
void mandelbrot_kernel_C(int const threshold,
                         double const realAxis[],      int const nreal,
                         double const imaginaryAxis[], int const nimag,
                         int atlas[]);

/** Mandelbrot iteration in a certain complex domain using the kernel function with C-like interface.
 *
 *  @param[in]  threshold      maximal number of iterations
 *  @param[in]  density        number of pixels in x- and y-direction
 *  @return 2d-vector[@p nreal][@p nimag]  with stopping iteration index for each pixel.
 *  @see mandelbrot
*/
std::vector<std::vector<int>> mandelbrot_split_C(int const threshold, int const density);


// simply   return a==b; would work but I would like to see differences
template <class T>
bool isEqual(std::vector<std::vector<T>> const &a, std::vector<std::vector<T>> const &b)
{
    bool bb=a.size()==b.size();
    for (size_t kx=0; kx<a.size(); ++kx)
    {
	auto const &ax=a[kx];
	auto const &bx=b[kx];
	bb = bb && ax.size()==bx.size();
	bb = bb && ax==bx;
	for (size_t ky=0; ky<ax.size(); ++ky)
	{
	    if (ax[ky]!=bx[ky]) 
	    {
		std::cout << "["<<kx<<","<<ky<<"] : (" << ax[ky] << "!=" << bx[ky]<< ")" << std::endl; 
	    }
	}
	
    }
    return bb;
}


/** Output operator for vector
 *  @param[in,out] s	output stream, e.g. @p cout
 *  @param[in]     v    vector
 *
 *  @return    output stream
*/
template <class T>
std::ostream& operator<<(std::ostream &s, std::vector<T> const &v)
{
    for (auto vp: v)
    {
        s << std::setprecision(8) << vp << "  ";
    }
    return s;
}


