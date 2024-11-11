import numpy as np
import matplotlib.pyplot as plt
# timing via time()
from time import time

# counts the number of iterations until the function diverges or
# returns the iteration threshold that we check until
def countIterationsUntilDivergent(c, threshold):
    z = complex(0, 0)
    for iteration in range(threshold):
        z = (z*z) + c

        if abs(z) > 4:
            break
    
    return iteration


# takes the iteration limit before declaring function as convergent and
# takes the density of the atlas
# create atlas, plot mandelbrot set, display set
def mandelbrot(threshold, density):
    # location and size of the atlas rectangle
    realAxis = np.linspace(-0.22, -0.219, density)
    imaginaryAxis = np.linspace(-0.70, -0.699, density)
    realAxisLen = len(realAxis)
    imaginaryAxisLen = len(imaginaryAxis)
    # 2-D array to represent mandelbrot atlas
    atlas = np.empty((realAxisLen, imaginaryAxisLen))

    # color each point in the atlas depending on the iteration count
    for iy in range(imaginaryAxisLen):
        for ix in range(realAxisLen):
            cx = realAxis[ix]
            cy = imaginaryAxis[iy]
            c = complex(cx, cy)

            atlas[ix, iy] = countIterationsUntilDivergent(c, threshold)
    
    return atlas


nd = 1000             # nd x nd domain
# time to party!!
t1 = time ( )
atlas = mandelbrot(120, nd)
tdiff = time ( ) - t1
print ('Operations completed after %g seconds' % ( tdiff ))

# plot and display mandelbrot set
plt.imshow(atlas.T, interpolation="nearest")
plt.show()

