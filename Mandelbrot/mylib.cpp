#include "mylib.h"
#include <complex>
#include <vector>           // vector
using namespace std;

// counts the number of iterations until the function diverges or
// returns the iteration threshold that we check until
//int countIterationsUntilDivergent(complex<double> const c, int const threshold)
//{
    //complex<double> z(0, 0);
    //int iteration;
    //for (iteration = 0; iteration < threshold; ++iteration) {
        //z = (z * z) + c;

        //if (abs(z) > 4) {
            //break;
        //}
    //}
    //return iteration;
//}


int countIterationsUntilDivergent(complex<double> const &c, int const threshold)
{
    complex<double> z(0.0, 0.0);
    int iteration = 0;
//    while(abs(z) <= 4 && iteration < threshold){ // profiling: 72% time in hypot
    while (norm(z) <= 4.0 * 4.0 && iteration < threshold) {
        z = (z * z) + c;
        ++iteration;
    }
    return iteration;
}


// takes the iteration limit before declaring function as convergent and
// takes the density of the atlas
// create atlas, plot mandelbrot set, display set
vector<vector<int>> mandelbrot(int const threshold, int const density)
{
    // location and size of the atlas rectangle
    vector<double> realAxis(density);
    vector<double> imaginaryAxis(density);
    // 2-D vector to represent mandelbrot atlas
    vector<vector<int>> atlas(realAxis.size(), vector<int>(imaginaryAxis.size()));

    for (int i = 0; i < density; ++i) {
        realAxis[i] = -0.22 + (i * (0.001 / density));
        imaginaryAxis[i] = -0.70 + (i * (0.001 / density));
    }

    // color each point in the atlas depending on the iteration count
    #pragma omp parallel for default(none) shared(realAxis,imaginaryAxis,atlas) firstprivate(threshold)
    for (int ix = 0; ix < realAxis.size(); ++ix) {
        for (int iy = 0; iy < imaginaryAxis.size(); ++iy) {
            complex<double> c(realAxis[ix], imaginaryAxis[iy]);
            atlas[ix][iy] = countIterationsUntilDivergent(c, threshold);
        }
    }

    return atlas;
}

// ---------------------------------------------------------------------------------------
void mandelbrot_kernel(int const threshold, vector<double> const &realAxis, vector<double> const &imaginaryAxis,
                       vector<vector<int>> &atlas)
{
    int const nreal = static_cast<int>(realAxis.size());
    int const nimag = static_cast<int>(imaginaryAxis.size());
// color each point in the atlas depending on the iteration count
    #pragma omp parallel for default(none) shared(realAxis,imaginaryAxis,atlas) firstprivate(threshold, nreal, nimag)
    for (int ix = 0; ix < nreal; ++ix) {
        for (int iy = 0; iy < nimag; ++iy) {
            complex<double> c(realAxis[ix], imaginaryAxis[iy]);
            atlas[ix][iy] = countIterationsUntilDivergent(c, threshold);
        }
    }
}

vector<vector<int>> mandelbrot_split(int const threshold, int const density)
{
    // location and size of the atlas rectangle
    vector<double> realAxis(density);
    vector<double> imaginaryAxis(density);
    // 2-D vector to represent mandelbrot atlas
    vector<vector<int>> atlas(realAxis.size(), vector<int>(imaginaryAxis.size()));

    for (int i = 0; i < density; ++i) {
        realAxis[i] = -0.22 + (i * (0.001 / density));
        imaginaryAxis[i] = -0.70 + (i * (0.001 / density));
    }
    mandelbrot_kernel(threshold, realAxis, imaginaryAxis, atlas);

    return atlas;
}

// ---------------------------------------------------------------------------------------
void mandelbrot_kernel_C(int const threshold,
                         double const realAxis[],      int const nreal,
                         double const imaginaryAxis[], int const nimag,
                         int atlas[])
{
// color each point in the atlas depending on the iteration count
    #pragma omp parallel for default(none) shared(realAxis,imaginaryAxis,atlas) firstprivate(threshold, nreal, nimag)
    for (int ix = 0; ix < nreal; ++ix) {
        for (int iy = 0; iy < nimag; ++iy) {
            complex<double> c(realAxis[ix], imaginaryAxis[iy]);
            atlas[ix*nimag+iy] = countIterationsUntilDivergent(c, threshold);
        }
    }
}

vector<vector<int>> mandelbrot_split_C(int const threshold, int const density)
{
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
    mandelbrot_kernel_C(threshold, realAxis.data(), nreal, imaginaryAxis.data(), nimag, atlas1d.data());
    
    //cout << "atlas1d: " << atlas1d << endl;
    // 1D--> 2D
    int idx=0;
    for (int ix = 0; ix < nreal; ++ix) {
        for (int iy = 0; iy < nimag; ++iy) {
            atlas[ix][iy] = atlas1d[idx++];
        }
    } 

    return atlas;
}
