//https://www.codeconvert.ai/python-to-c++-converter

#include <algorithm>
#include <chrono>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>
//#include <SFML/Graphics.hpp>

using namespace std;

// counts the number of iterations until the function diverges or
// returns the iteration threshold that we check until
int countIterationsUntilDivergent(complex<double> c, int threshold) {
    complex<double> z(0, 0);
    int iteration;
    for (iteration = 0; iteration < threshold; ++iteration) {
        z = (z * z) + c;

        if (abs(z) > 4) {
            break;
        }
    }
    return iteration;
}

// takes the iteration limit before declaring function as convergent and
// takes the density of the atlas
// create atlas, plot mandelbrot set, display set
vector<vector<int>> mandelbrot(int threshold, int density) {
    // location and size of the atlas rectangle
    vector<double> realAxis(density);
    vector<double> imaginaryAxis(density);
    for (int i = 0; i < density; ++i) {
        realAxis[i] = -0.22 + (i * (0.001 / density));
        imaginaryAxis[i] = -0.70 + (i * (0.001 / density));
    }

    // 2-D vector to represent mandelbrot atlas
    vector<vector<int>> atlas(realAxis.size(), vector<int>(imaginaryAxis.size()));

    // color each point in the atlas depending on the iteration count
#pragma omp parallel for    
    for (size_t ix = 0; ix < realAxis.size(); ++ix) {
        for (size_t iy = 0; iy < imaginaryAxis.size(); ++iy) {
            complex<double> c(realAxis[ix], imaginaryAxis[iy]);
            atlas[ix][iy] = countIterationsUntilDivergent(c, threshold);
        }
    }

    return atlas;
}

int main() {
    int nd = 1000; // nd x nd domain
    auto t1 = chrono::high_resolution_clock::now();
    auto atlas = mandelbrot(120, nd);
    auto tdiff = chrono::high_resolution_clock::now() - t1;
    auto duration = chrono::duration_cast<chrono::milliseconds>(tdiff).count();
    cout << "Operations completed after " << duration / 1000.0 << " seconds" << endl;

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

