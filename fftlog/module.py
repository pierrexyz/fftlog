import os, sys
from fftlog.config import get_jax_enabled
global is_jax
is_jax = get_jax_enabled()
if is_jax:
    from jax import jit, vmap, disable_jit
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True) 
    import jax.numpy as numpy
    from jax.numpy import load, asarray, array, ndarray, conj, ones, tan, log, logspace, swapaxes, empty, linspace, arange, delete, where, pi, cos, sin, log, exp, sqrt, concatenate, ones, zeros, real, where, einsum
    from jax.numpy.fft import rfft, fft, fftshift, rfftfreq, fftfreq, irfft, ifft, ifftshift
    from fftlog.jax_special import interp1d
    from scipy.special import gamma, loggamma

else:
    import numpy
    from numpy import load, asarray, array, ndarray, conj, ones, tan, log, logspace, swapaxes, empty, array, linspace, arange, delete, where, pi, cos, sin, log, exp, sqrt, concatenate, ones, zeros, real, where, einsum
    from numpy.fft import rfft, fft, fftshift, rfftfreq, fftfreq, irfft, ifft, ifftshift
    from scipy.interpolate import interp1d

from scipy.special import gamma, loggamma