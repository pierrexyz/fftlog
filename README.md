# FFTLog

A Python implementation of FFTLog for fast logarithmic FFT transforms, developed for [PyBird](https://github.com/pierrexyz/pybird).

## Installation

### Basic installation (recommended)
```bash
pip install fftlog-lss
```

### With JAX acceleration
```bash
pip install "fftlog-lss[jax]"
```

### From source
```bash
git clone https://github.com/pierrexyz/fftlog.git
cd fftlog
pip install --editable .
```

### From source with JAX
```bash
git clone https://github.com/pierrexyz/fftlog.git
cd fftlog
pip install --editable ".[jax]"
```

## Dependencies

### Required
- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0

### Optional (for JAX acceleration)
- jax >= 0.4.0
- jaxlib >= 0.4.0
- interpax >= 0.3.0

## Quickstart

### Basic Usage (without JAX)

```python
import numpy as np
from scipy.stats import lognorm
from fftlog import FFTLog

# Create test function (power spectrum)
k = np.logspace(-4, 0, 200)
pk = lognorm.pdf(k, 2.1)

# Initialize FFTLog
fft = FFTLog(
    Nmax=512,       # Number of points
    xmin=1e-5,      # Minimum k value
    xmax=1e3,       # Maximum k value
    bias=-0.1,      # Bias parameter
    complex=False,  # Use real FFT
    window=0.1      # Anti-aliasing window
)

# FFTLog decomposition and reconstruction
pk_reconstructed = fft.rec(k, pk, k)

# Spherical Bessel Transform
s = np.arange(1., 1e3, 5.)
xi = fft.sbt(k, pk, s)  # Fast O(N log N) transform
```

### With JAX Acceleration

```python
import numpy as np
from scipy.stats import lognorm
from fftlog import FFTLog
from fftlog.config import set_jax_enabled
from jax import jit
import jax.numpy as jnp

# Enable JAX mode
set_jax_enabled(True)

# Create test function
k = np.logspace(-4, 0, 200)
pk = lognorm.pdf(k, 2.1)

# Initialize FFTLog (same as above)
fft = FFTLog(
    Nmax=512, xmin=1e-5, xmax=1e3, 
    bias=-0.1, complex=False, window=0.1
)

# Convert to JAX arrays and JIT compile
k_jax, pk_jax = jnp.array(k), jnp.array(pk)
get_coef_jit = jit(fft.Coef)

# Now much faster for repeated calls
coefficients = get_coef_jit(k_jax, pk_jax)
```

## Features

- Fast logarithmic FFT transforms
- Support for both real and complex transforms
- Spherical Bessel transforms
- Anti-aliasing windows
- Optional JAX acceleration for GPU/TPU support

## Documentation

For more detailed examples and documentation, see the notebooks in the `notebooks/` directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



