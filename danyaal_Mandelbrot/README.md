# Mandelbrot example:
Some experiments for acceleration in the complex domain $\Omega=[-0.22, -0.219]\times [-0.70, -0.699]$ with nd=1000 points in each direction.
Double precision is used.
The iteration $z = z*z+c$ with $c\in\Omega$ is stopped when $|z|>4$ (resp. $|z|^2>4*4$) is reached or at 120 iterations.

Login to [jupyter notebook](https://imsc.uni-graz.at/jupyter/hub/login) mentioned as meph(ip).
 
## what to install
The experiments require some python packages: matplotlib, numpy, llvmlite, numba.

I created a new virtual environment on my desktop-PC (Ubuntu 24.04)
    `python3 -m venv ~/py_envs`
Active that environment prior to each session 
    `source ~/py_envs/bin/activate`
Install packages for this environment (once)
    `python3 -m pip install matplotlib`
    `python3 -m pip install numpy llvmlite numba`

### mandelbrot.py
 - original code by  [Danyaal Rangwala](https://github.com/danyaal/mandelbrot/blob/master/mandelbrot.py)


### gh_org.py
 - restructured original code includes timing
 - math125: 7.2 sec                                       **Baseline**
 - math125:11.7 sec avoiding abs()
 - math125: 7.3 sec columnwise access in `gh_org_cw.py`
 
 - mephisto: 11.1 sec (w/o compilation) and avoiding abs()

### gh_jit.py
 - using [jit](https://numba.readthedocs.io/en/stable/user/performance-tips.html)-compiler
 - math125:  0.96 sec (incl. compilation)
 - math125:  0.51 sec (w/o compilation)                    **Speedup 14**
 - math125:  0.51 sec (w/o compilation) and columnwise access in `gh_jit_cw.py`
 - math125:  0.17 sec (w/o compilation) and avoiding abs() **Speedup 42**
 
 - mephisto: 0.24 sec (w/o compilation) and avoiding abs() **Speedup 46**
 
### gh_njit.py
 - using [njit](https://numba.readthedocs.io/en/stable/user/parallel.html) for parallelization 
 - math125:  0.064 sec (w/o compilation); 8 cores
 - math125:  0.084 sec (w/o compilation); 8 cores and columnwise access in `gh_njit_cw.py`
 - math125:  0.017 sec (w/o compilation); 8 cores and avoiding abs()   **Speedup  423**
 
 - mephisto: 0.0116 sec (w/o compilation) and avoiding abs()           **Speedup  957**

 
### gh_numbaCUDA.ipynb
timings (w/o compilation) and avoiding abs()
 - meph(ip): 19.0 sec   (python)          **Baseline**  (lower clock rate on server, overhead notebook?)
 - meph(ip): 0.24 sec   (@jit)            **Speedup  79** 
 - meph(ip): 0.0053 sec (@njit)           **Speedup 3585**
 - meph(ip): 0.0037 sec (@cuda.jit)       **Speedup 5135** 


### Examples/Mandelbrot with C++  and nd=1000
 - math125:   0.26   sec; avoiding abs() with OpenMP and  1 thread
 - math125:   0.022  sec; avoiding abs() with OpenMP and 16 threads
 - math125:   0.013  sec; avoiding abs() with CUDA on GTX1660
 
 - mephisto:  0.35   sec; avoiding abs() with OpenMP and   1 thread             **Baseline**
 - mephisto:  0.025  sec; avoiding abs() with OpenMP and  16 threads
 - mephisto:  0.014  sec; avoiding abs() with OpenMP and  32 threads
 - mephisto:  0.013  sec; avoiding abs() with OpenMP and  64 threads 
 - mephisto:  0.010  sec; avoiding abs() with OpenMP and 128 threads            **Speedup  35**
 - mephisto:  0.0042 sec; avoiding abs() with CUDA on A100 (incl. data handling)**Speedup  84**
 - mephisto:  0.0007 sec; avoiding abs() with CUDA on A100 (pure GPU time)      **Speedup 500**

### Examples/Mandelbrot with C++  and nd=10000
 - mephisto: 34    sec; avoiding abs() with OpenMP and   1 thread              **Baseline**
 - mephisto:  1.5  sec; avoiding abs() with OpenMP and  32 threads
 - mephisto:  1.0  sec; avoiding abs() with OpenMP and  64 threads 
 - mephisto:  0.7  sec; avoiding abs() with OpenMP and 128 threads 
 - mephisto:  0.56 sec; avoiding abs() with OpenMP and 256 threads             **Speedup  61**
 - mephisto:  0.6  sec; avoiding abs() with CUDA on A100 (incl. data handling)
 - mephisto:  0.06 sec; avoiding abs() with CUDA on A100 (pure GPU time)       **Speedup 566**


## GPU and Python
Excellent Mandelbrot example by [Jérémy Lesuffleur](https://www.kaggle.com/code/jlesuffleur/cuda-accelerating-mandelbrot-fractal-with-gpu) using `@cuda.jit`.
His code `kaggle_numbaCUDA.ipnyb` runs (with minor modifications) on [jupyter hub](https://imsc.uni-graz.at/jupyter/hub/login).
The appropriate python cdoe is `kaggle_numbaCUDA.py`.

 - math125: CPU 10.4 sec --> CPU(@njit) 0.0136 sec   **Speedup 765**  but no further speedup with GPU(@cuda.jit)
   | python     | CPU (python) | CPU (@jit) | CPU (@njit) | GPU (@cuda.jit) |
   |------------| ------------ | ---------- | ----------- | --------------- |
   | time msec] |  10400       |    176     |    13.6     |    14.4         |
   | SpeedUp    |      1       |     59     |   765       |   722           |
   
  - mephisto: CPU 14.6 sec --> CPU(@njit) 0.0079 sec   **Speedup 1848** 
   | python     | CPU (python) | CPU (@jit) | CPU (@njit) | GPU (@cuda.jit) |
   |------------| ------------ | ---------- | ----------- | --------------- |
   | time msec] |  14600       |    236     |     7.9     | IR version !=   |
   | SpeedUp    |      1       |     62     |  1848       | -----           | 
   
  - mephisto: CPU 16.9 sec --> CPU(@njit) 0.0077 sec   **Speedup 2195**   and with GPU(@cuda.jit)   **Speedup 5827**
   | ipynp      | CPU (python) | CPU (@jit) | CPU (@njit) | GPU (@cuda.jit) |
   |------------| ------------ | ---------- | ----------- | --------------- |
   | time msec] |   16900      |    246     |     7.7     |      2.9        |
   | SpeedUp    |      1       |     69     |  2195       |   5827          | 
