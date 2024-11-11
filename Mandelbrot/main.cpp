//https://www.codeconvert.ai/python-to-c++-converter
#include "mylib.h"
#include "mylib_gpu.h"
#include "timing.h"
#include <algorithm>
#include <chrono>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>
//#include <SFML/Graphics.hpp>

using namespace std;

int main() {
    int const nd = 10000;    // nd x nd domain
    //int const nd = 10;    // nd x nd domain
    int const maxiter=120;  // max. nuber of iterations
    cout.precision(2);

    // fill caches and reference solution
    auto const atlas_ref = mandelbrot(maxiter, nd);
    
    {
    tic();
    auto atlas = mandelbrot(maxiter, nd);
    cout << "CPU 0: Operations completed after " << toc() << " seconds" << endl;
    if ( !isEqual(atlas_ref,atlas) )  cout << "Unequal atlas2" << endl;
    }

    {
    tic();
    auto atlas = mandelbrot_split(maxiter, nd);
    cout << "CPU 1: Operations completed after " << toc() << " seconds" << endl;
    if ( !isEqual(atlas_ref,atlas) )  cout << "Unequal atlas2" << endl;
    }
    
    {
    tic();
    auto atlas = mandelbrot_split_C(maxiter, nd);
    cout << "CPU 2: Operations completed after " << toc() << " seconds" << endl;
    if ( !isEqual(atlas_ref,atlas) )  cout << "Unequal atlas3" << endl;
    }
    
#if defined __NVCC__
    { // init GPU
    auto atlas = mandelbrot_split_GPU(maxiter, nd);
    tic();
    atlas = mandelbrot_split_GPU(maxiter, nd);
    cout << "GPU A: Operations completed after " << toc() << " seconds" << endl;
    if ( !isEqual(atlas_ref,atlas) )  cout << "Unequal atlas4" << endl;
    }
#endif    

    //// plot and display mandelbrot set using SFML
    //sf::RenderWindow window(sf::VideoMode(nd, nd), "Mandelbrot Set");
    //sf::Image image;
    //image.create(nd, nd, sf::Color::Black);
    
    //for (int x = 0; x < nd; ++x) {
        //for (int y = 0; y < nd; ++y) {
            //int iterations = atlas[x][y];
            //int colorValue = static_cast<int>(255.0 * iterations / 120);
            //image.setPixel(x, y, sf::Color(colorValue, colorValue, colorValue));
        //}
    //}

    //sf::Texture texture;
    //texture.loadFromImage(image);
    //sf::Sprite sprite(texture);

    //while (window.isOpen()) {
        //sf::Event event;
        //while (window.pollEvent(event)) {
            //if (event.type == sf::Event::Closed)
                //window.close();
        //}

        //window.clear();
        //window.draw(sprite);
        //window.display();
    //}

    return 0;
}

