from fftlog.module import *
from fftlog.utils import CoefWindow,  MPC

class FFTLog(object):
    """
    A class implementing the FFTLog algorithm.
    """

    def __init__(self, **kwargs):
        self.Nmax = kwargs['Nmax']
        self.xmin = kwargs['xmin']
        self.xmax = kwargs['xmax']
        self.bias = kwargs['bias']
        self.is_window = True if 'window' in kwargs else False
        self.is_complex = kwargs['complex'] if 'complex' in kwargs else False
        self.is_hankl = kwargs['hankl'] if 'hankl' in kwargs else False

        self.dx = log(self.xmax / self.xmin) / (self.Nmax - 1.)
        ii = arange(self.Nmax)
        self.x = self.xmin * exp(ii * self.dx)
        self.xpb = exp(-self.bias * ii * self.dx)

        self.freqs = fftshift(fftfreq(self.Nmax, d=self.dx)) # full spectrum for complex FFT, shifted such that it runs on [-N/2, N/2-1] instead of [0, N] ( = arange(-N//2, N/2) / (self.Nmax * self.dx) )
        # self.freqs = fftfreq(self.Nmax, d=self.dx) # full spectrum for complex FFT running on [0, N] 
        self.Pow = self.bias + 1j * 2 * pi * self.freqs # complex powers, to be used with explicit sum

        if self.is_complex: freqs = fftfreq(self.Nmax, d=self.dx) # self.freqs # full-spectrum for complex FFT running on [0, N]
        else: freqs = rfftfreq(self.Nmax, d=self.dx) # half-spectrum (positive frequencies only) for real FFT running on [0, N/2]
        self._Pow = self.bias + 1j * 2 * pi * freqs # real or complex powers, to be used with irfft or ifft, respectively

        self.xminPow_normN = exp(-self.Pow*log(self.xmin)) / float(self.Nmax)

        if self.is_window:
            if kwargs['window'] is None: kwargs['window'] = 0. # setting no window
            if kwargs['window'] < 0. and kwargs['window'] > 1.: raise Exception('Set option \'window\' between 0 (no window) and 1 (maximal anti-aliasing)') 
            if self.is_complex: self.W = fftshift(CoefWindow(self.Nmax, window=kwargs['window'])) # complex fft, symmetric anti-aliasing window (shifted in frequencies to match the natural ordering of the fft)
            else: self.W = CoefWindow(self.Nmax, left=False, right=True, window=kwargs['window'])[self.Nmax//2-1:] # real fft, anti-aliasing window (only on the right) 

        if self.is_hankl: 
            self.ells = kwargs['ells'] if 'ells' in kwargs else [0,] # [0, 2, 4]
            m_y = arange(-self.Nmax // 2, self.Nmax // 2)
            self.y = self.x[self.Nmax // 2]**-1 * exp(-self.dx * m_y)[::-1] # output of the convolution: reversed log-spaced grid centred around inverse of midpoint x
            self.shifted_idx = fftshift(m_y) # to order the output of irfft as self.y
            self.u_m = array([1j**ell * MPC(ell, -.5*self._Pow) for ell in self.ells]) # matrices of the spherical-Bessel transform from Ps to Cf
            self.v_m = 8.*pi**3 * array([(-1j)**ell * MPC(ell, -.5*self._Pow) for ell in self.ells]) # matrices of the spherical-Bessel transform from Cf to Ps
            

    def Coef(self, xin, f, mode='interp', extrap='extrap', to_sum=True):

        if mode == 'exact':
            fx = 1. * asarray(f)
        elif mode == 'interp':
            if extrap == 'extrap': 
                iloglog = interp1d(log(xin), log(f), axis=-1, kind='linear', bounds_error=False, fill_value='extrapolate')
                fx = exp(iloglog(log(self.x)))
            elif extrap == 'padding': 
                ifx = interp1d(xin, f, axis=-1, kind='cubic', bounds_error=False, fill_value=0.) 
                fx = ifx(self.x)

        fx = fx * self.xpb
        
        if self.is_complex: c_m = fft(fx, axis=-1) # runs on [-N/2, N/2-1]
        else: c_m = rfft(fx, axis=-1)
        
        if self.is_window: c_m = c_m * self.W # anti-aliasing window by filtering high-frequencies near the Nyquist
        if to_sum: # the output coefficients c_m are now defined such that the input function = \sum_m c_m x^Pow_m with m running from [-N/2,N/2-1]; put to False when using ifft / irfft  
            if self.is_complex: c_m = fftshift(c_m, axes=-1) # runs on [-N/2, N/2-1]
            else: c_m = concatenate((conj(c_m[...,::-1][...,:-1]), c_m[...,:-1]), axis=-1) # For the real FFT, replicating the spectrum with the symmetric conjugate (and we remove the double-counted elements, such that the sum goes runs on [-N/2, N/2-1])
            # if not self.is_complex: c_m = concatenate((c_m, conj(c_m[...,-2:0:-1])), axis=-1) # # For the real FFT, replicating the spectrum with conjugate & reverse, skipping Nyquist and m=0 to avoid duplicates
            c_m = c_m * self.xminPow_normN # normalisation

        return c_m

    def sumCoefxPow(self, xin, f, x):
        """Reconstruction of input function decomposed with the FFTLog through O(N^2)-sum"""
        c_m = self.Coef(xin, f)
        return array([real(sum(c_m * xi**self.Pow)) for xi in x])

    def rec(self, xin, f, x=None, mode='interp', extrap='extrap'):
        """Reconstruction of input function decomposed with the FFTLog through O(NlogN)-iFFT
        """
        c_m = self.Coef(xin, f, mode=mode, extrap=extrap, to_sum=False) 
        if self.is_complex: rf = ifft(c_m, axis=-1) 
        else: rf = irfft(c_m, axis=-1) 
        rf = rf / self.xpb
        if x is not None: 
            if extrap == 'extrap': 
                iloglog = interp1d(log(self.x), log(rf), axis=-1, kind='linear', bounds_error=False, fill_value='extrapolate')
                rf = exp(iloglog(log(x)))
            elif extrap == 'padding': 
                ifx = interp1d(self.x, rf, axis=-1, kind='cubic', bounds_error=False, fill_value=0.) 
                rf = ifx(x)
        return rf

    def sbt(self, xin, f,  y=None, kernel=None, f2c=True, sum_ell=False, return_y=False, einsum_optimize=False, mode='interp', extrap='extrap'):
        """Fast spherical Bessel tranform
        """
        c_m = self.Coef(xin, f, mode=mode, extrap=extrap, to_sum=False) 
        
        if kernel is None: 
            if f2c: u_m = self.u_m # fourier to configuration space
            else: u_m = self.v_m # configuration to fourier space
        else: u_m = kernel
        if sum_ell: cu_m = einsum('...lm,lm->...lm', c_m, u_m, optimize=einsum_optimize)
        else: cu_m = einsum('...m,lm->...lm', c_m, u_m, optimize=einsum_optimize)
        if self.is_complex: h_m = ifft(cu_m, axis=-1) / self.xpb
        else: h_m = irfft(cu_m, axis=-1) / self.xpb
        f_y = h_m[...,self.shifted_idx][...,::-1] # Apply same shift and reverse as for y
        f_y = f_y * self.y**-3.
        if y is None: y = self.y
        else:
            ifx = interp1d(self.y, f_y, axis=-1, kind='cubic', bounds_error=False, fill_value=0.) 
            f_y = ifx(y)
        if return_y: return y, f_y
        else: return f_y 




        

