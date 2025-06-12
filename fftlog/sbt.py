from fftlog.module import *
from fftlog.fftlog import FFTLog
from fftlog.utils import MPC

class SBT(object):

    def __init__(self, to_fourier=True, ells=[0, 2, 4], **kwargs):
        self.ells = ells
        if to_fourier: self.set_c2f(**(kwargs or {}))
        else: self.set_f2c(**(kwargs or {}))

    def set_c2f(self, k=None, smin=1e-1, smax=1e5, NFFT=512, bias=-.01, mode='interp', extrap='padding', window=.2):
        """configuration space to Fourier space"""
        if k is None: k = arange(1e-3, 1., 0.005)
        self.x = k # output k
        self.fftsettings = dict(Nmax=NFFT, xmin=smin, xmax=smax, bias=bias, window=window)
        self.fft = FFTLog(**self.fftsettings)
        self.fft.mode, self.fft.extrap = mode, extrap
        self.M = 8.*pi**3 * array([(-1j)**ell * MPC(ell, -.5*self.fft.Pow) for ell in self.ells]) # matrices of the spherical-Bessel transform from Cf to Ps.
        self.xPow = exp(einsum('n,x->nx', -self.fft.Pow - 3., log(self.x))) 

    def set_f2c(self, s=None, kmin=1e-4, kmax=1e3, NFFT=512, bias=.01, mode='interp', extrap='extrap', window=.2):
        """Fourier space to configuration space"""
        if s is None: s = arange(1., 1e3, 5.) # logspace(-3, 3, 200) #
        self.x = s # output s
        self.fftsettings = dict(Nmax=NFFT, xmin=kmin, xmax=kmax, bias=bias, window=window) 
        self.fft = FFTLog(**self.fftsettings)
        self.fft.mode, self.fft.extrap = mode, extrap
        self.M = array([1j**ell * MPC(ell, -.5*self.fft.Pow) for ell in self.ells])
        self.xPow = exp(einsum('n,x->nx', -self.fft.Pow - 3., log(self.x))) 

    def get_transform(self, xin, f, sum_ell=False):
        Coef = self.fft.Coef(xin, f, mode=self.fft.mode, extrap=self.fft.extrap)
        if sum_ell:
            CoefxPow = einsum('...ln,nx->...lnx', Coef, self.xPow)
            return real(einsum('...lnx,ln->...lx', CoefxPow, self.M)) 
        else:
            CoefxPow = einsum('...n,nx->...nx', Coef, self.xPow)
            return real(einsum('...nx,ln->...lx', CoefxPow, self.M)) 

