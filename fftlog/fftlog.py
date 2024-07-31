from fftlog.module import *
from fftlog.utils import CoefWindow

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

        self.dx = log(self.xmax / self.xmin) / (self.Nmax - 1.)
        self.x = array([self.xmin * exp(i * self.dx) for i in range(self.Nmax)])
        self.xpb = array([exp(-self.bias * i * self.dx) for i in range(self.Nmax)])
        self.Pow = array([self.bias + 1j * 2. * pi * i / (self.Nmax * self.dx) for i in arange(-self.Nmax//2, self.Nmax//2+1)])

        if 'window' in kwargs:
            self.window = kwargs['window']
            self.W = CoefWindow(self.Nmax, window=self.window)
        else:
            self.window = None


    def Coef(self, xin, f, extrap='extrap'):

        if extrap == 'extrap':
            iloglog = interp1d(log(xin), log(f), axis=-1, kind='linear', bounds_error=False, fill_value='extrapolate')
            fx = exp(iloglog(log(self.x)))

        elif extrap == 'padding': 
            ifunc = interp1d(xin, f, axis=-1, kind='cubic')
            def f(x): return where((x < xin[0]) | (xin[-1] < x), 0.0, ifunc(x))
            if is_jax: fx = vmap(lambda x: f(x))(self.x)
            else: fx = f(self.x)

        fx = fx * self.xpb
        tmp = rfft(fx, axis=-1)
        Coef = concatenate((conj(tmp[...,::-1][...,:-1]), tmp), axis=-1)
        Coef = Coef * exp(-self.Pow*log(self.xmin)) / float(self.Nmax)

        if self.window:
            Coef = Coef * self.W

        else:
            if is_jax:
                Coef = Coef.at[0].divide(2.).at[self.Nmax].divide(2.)
            else:
                Coef[...,0] /= 2.
                Coef[...,self.Nmax] /= 2.

        return Coef

    def sumCoefxPow(self, xin, f, x):
        Coef = self.Coef(xin, f)
        return array([real(sum(Coef * xi**self.Pow)) for xi in x])