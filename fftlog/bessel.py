from fftlog.module import *

def MPC(l, pn):
    """ matrix for spherical bessel transform from power spectrum to correlation function """
    # return np.pi**-1.5 * 2.**(-2. * pn) * gamma(1.5 + l / 2. - pn) / gamma(l / 2. + pn)
    return pi**-1.5 * 2.**(-2. * pn) * exp(loggamma(1.5 + l / 2. - pn) - loggamma(l / 2. + pn))