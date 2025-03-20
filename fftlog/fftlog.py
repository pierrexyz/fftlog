from fftlog.module import *
from fftlog.utils import CoefWindow

import interpax

class FFTLog(object):
    """
    A class implementing the FFTLog algorithm.

    Attributes
    ----------
    Nmax : int, optional
        maximum number of points used to discretize the function
    xmin : float, optional
        minimum of the function to transform
    xmax : float, optional
        maximum of the function to transform
    bias : float, optional
        power by which we modify the function as x**bias * f

    Methods
    -------
    setx()
        Calculates the discrete x points for the transform

    setPow()
        Calculates the power in front of the function

    Coef()
        Calculates the single coefficients

    sumCoefxPow(xin, f, x, window=1)
        Sums over the Coef * Pow reconstructing the input function
    """

    def __init__(self, **kwargs):
        self.Nmax = kwargs['Nmax']
        self.xmin = kwargs['xmin']
        self.xmax = kwargs['xmax']
        self.bias = kwargs['bias']
        self.is_complex = kwargs['complex'] if 'complex' in kwargs else False

        self.dx = log(self.xmax / self.xmin) / (self.Nmax - 1.)
        i = arange(self.Nmax)
        self.x = self.xmin * exp(i * self.dx)
        self.xpb = exp(-self.bias * i * self.dx)
        if self.is_complex: 
            self.Pow = self.bias + 1j * 2 * pi * fftshift(fftfreq(self.Nmax, d=self.dx)) 
        else: 
            i = arange(-self.Nmax//2, self.Nmax//2+1)
            self.Pow = self.bias + 1j * 2. * pi * i / (self.Nmax * self.dx)

        if 'window' in kwargs:
            self.window = kwargs['window']
            if self.is_complex: self.W = CoefWindow(self.Nmax-1, window=self.window)
            else: self.W = CoefWindow(self.Nmax, window=self.window)
        else:
            self.window = None

    def Coef(self, xin, f, mode='interp', extrap='extrap'):

        if mode == 'exact':
            fx = 1. * asarray(f)
        
        elif mode == 'interp':
            
            if extrap == 'extrap':
                iloglog = interp1d(log(xin), log(f), axis=-1, kind='linear', bounds_error=False, fill_value='extrapolate')
                fx = exp(iloglog(log(self.x)))
            
            elif extrap == 'padding': 
                # f_mod = swapaxes(f, 0, 1)
                # f_interp = interpax.interp1d(self.x, xin, f_mod, extrap = True)
                # fx = einsum('N, Nk -> kN', f_01, f_interp)
                ifx = interp1d(xin, f, axis=-1, kind='cubic', bounds_error=False, fill_value=0.)
                fx = ifx(self.x)
                # mask_01 =  where((self.x < xin[0]) | (xin[-1] < self.x), 0.0, 1.)
                # fx = einsum('n, nx -> xn', mask_01, f_interp)

        fx = fx * self.xpb
        
        if self.is_complex: 
            Coef = fftshift(fft(fx, axis=-1), axes=-1)
        else: 
            tmp = rfft(fx, axis=-1)
            Coef = concatenate((conj(tmp[...,::-1][...,:-1]), tmp), axis=-1) 
        
        Coef = Coef * exp(-self.Pow*log(self.xmin)) / float(self.Nmax)

        if self.window:
            Coef = Coef * self.W
        elif not self.is_complex: # Complex goes from [-N/2, N/2-1], while real goes from [-N/2, N/2], so we halve the double-counted -N/2 = N/2 elements
            if is_jax:
                Coef = Coef.at[0].divide(2.).at[-1].divide(2.)
            else:
                Coef[...,0] /= 2.
                Coef[...,-1] /= 2.

        return Coef

    def sumCoefxPow(self, xin, f, x):
        Coef = self.Coef(xin, f)
        return array([real(sum(Coef * xi**self.Pow)) for xi in x])
