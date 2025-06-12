from fftlog.module import *
from fftlog.fftlog import FFTLog
from fftlog.utils import MPC

class Hankl(object):
	
	def __init__(self, to_fourier=True, ells=[0, 2, 4], **kwargs):
        self.ells = ells
        if to_fourier: self.set_c2f(**(kwargs or {}))
        else: self.set_f2c(**(kwargs or {}))

        self.y = 

    def get(self, xin, f,  y=None, mode='interp', extrap='extrap'):
        """Fast Hankel transform"""
        c_m = self.Coef(xin, f, mode=mode, extrap=extrap, to_sum=False) 
        h_m = irfft(c_m * self.u_m, axis=-1) / self.xpb
        if x is not None: 
            if extrap == 'extrap': 
                iloglog = interp1d(log(self.y), log(h_m), axis=-1, kind='linear', bounds_error=False, fill_value='extrapolate')
                h_m = exp(iloglog(log(y)))
            elif extrap == 'padding': 
                h_m = interp1d(self.y, h_m, axis=-1, kind='cubic', bounds_error=False, fill_value=0.) 
                h_m = ifx(y)
        return h_m
