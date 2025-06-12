from fftlog.module import *

def MPC(l, pn):
    """ matrix for spherical bessel transform from power spectrum to correlation function """
    # return pi**-1.5 * 2.**(-2. * pn) * gamma(1.5 + l / 2. - pn) / gamma(l / 2. + pn)
    return pi**-1.5 * 2.**(-2. * pn) * exp(loggamma(1.5 + l / 2. - pn) - loggamma(l / 2. + pn))

def CoefWindow(N, window=1., left=True, right=True):
    """ FFTLog auxiliary function: window sending the FFT coefficients to 0 at the edges. Adapted from fast-pt """

    if is_jax:
        n = arange(-N // 2, N // 2)
        if window == 1:
            n_cut = N // 2
        else:
            n_cut = int(window * N // 2.)

        n_right = n[-1] - n_cut
        n_left = n[0] + n_cut

        # Compute the masks for left and right windows directly
        mask_left = n < n_left
        mask_right = n > n_right

        # Compute theta values across the entire range, then mask
        theta_right = (n[-1] - n) / (n[-1] - n_right - 1)
        theta_left = (n - n[0]) / (n_left - n[0] - 1)

        # Apply windowing functions, ensuring operations are masked appropriately
        W_left = where(mask_left, theta_left - 1 / (2. * pi) * sin(2 * pi * theta_left), 1)
        W_right = where(mask_right, theta_right - 1 / (2. * pi) * sin(2 * pi * theta_right), 1)

        # Combine the windowing functions
        # Since the middle part is always 1, we can directly multiply W_left and W_right
        W = W_left * W_right

        return W

    else:
        n = arange(-N // 2, N // 2)
        if window == 1:
            n_cut = N // 2
        else:
            n_cut = int(window * N // 2.)

        n_right = n[-1] - n_cut
        n_left = n[0] + n_cut

        n_r = n[n[:] > n_right]
        n_l = n[n[:] < n_left]

        theta_right = (n[-1] - n_r) / float(n[-1] - n_right - 1)
        theta_left = (n_l - n[0]) / float(n_left - n[0] - 1)

        W = ones(n.size)
        if right: W[n[:] > n_right] = theta_right - 1 / (2. * pi) * sin(2 * pi * theta_right)
        if left: W[n[:] < n_left] = theta_left - 1 / (2. * pi) * sin(2 * pi * theta_left)

        return W