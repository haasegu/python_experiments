#  https://www.kaggle.com/code/jlesuffleur/cuda-accelerating-mandelbrot-fractal-with-gpu

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
#%matplotlib inline

##
# def mandelbrot(xpixels=600, maxiter=100, xmin=-2.6, xmax=1.85, ymin=-1.25, ymax=1.25):
def mandelbrot(xpixels=1000, maxiter=120, xmin=-0.22, xmax=-0.219, ymin=-0.70, ymax=-0.699):       # GH

    # Compute ypixels so the image is not stretched (1:1 ratio)
    ypixels = round(xpixels / (xmax-xmin) * (ymax-ymin))
    
    # Initialisation of result to 0
    mat = np.zeros((xpixels, ypixels))
    
    # Looping through pixels
    for x in range(xpixels):
        for y in range(ypixels):
            
            # Mapping pixel to C
            creal = xmin + x / xpixels * (xmax - xmin)
            cim = ymin + y / ypixels * (ymax - ymin)
            
            # Initialisation of C and Z
            c = complex(creal, cim)
            z = complex(0, 0)
            
            # Mandelbrot iteration
            for n in range(maxiter):
                z = z*z + c
                # If unbounded: save iteration count and break
                # if z.real*z.real + z.imag*z.imag > 4.0:
                if z.real*z.real + z.imag*z.imag > 4.0*4.0:               # GH
                    # Smooth iteration count
                    # mat[x,y] = n + 1 - math.log(math.log(abs(z*z+c)))/math.log(2)
                    mat[x,y] = n + 1          # GH
                    break
                # Otherwise: leave it to 0    # comment by GH: I set usually to maxiter in that case
    return(mat)
    
##
def draw_image(mat, cmap='inferno', powern=0.5, dpi=72):
    ## Value normalization
    # Apply power normalization, because number of iteration is 
    # distributed according to a power law (fewer pixels have 
    # higher iteration number)
    mat = np.power(mat, powern)
    
    # Colormap: set the color the black for values under vmin (inner points of
    # the set), vmin will be set in the imshow function
    # new_cmap = copy.copy(cm.get_cmap(cmap))  # depricated
    # new_cmap = copy.copy(cm._colormaps[cmap])    # GH
    new_cmap = copy.copy(plt.get_cmap())
  
    new_cmap.set_under('black')
    
    ## Plotting image
    
    # Figure size
    plt.figure(figsize=(mat.shape[0]/dpi, mat.shape[1]/dpi))
    
    # Plotting mat with cmap
    # vmin=1 because smooth iteration count is always > 1
    # We need to transpose mat because images use row-major
    # ordering (C convention)
    # origin='lower' because mat[0,0] is the lower left pixel
    plt.imshow(mat.T, cmap=new_cmap, vmin=1, origin = 'lower')
    
    # Remove axis and margins
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')

##

# Running and plotting result
print ("start")
t1 = time.time ( )
mat = mandelbrot()
tdiff = time.time ( ) - t1
print("end")
print ('Operations completed after %g seconds' % ( tdiff ))

# mat = mandelbrot()
draw_image(mat)


# timeit -n 10 -r 3 mandelbrot()

