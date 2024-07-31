is_jax_enabled = False  # Default setting

def set_jax_enabled(value: bool):
    global is_jax_enabled
    is_jax_enabled = value

    import importlib, fftlog.fftlog, fftlog.module
    importlib.reload(fftlog.module)
    importlib.reload(fftlog.fftlog)

def get_jax_enabled():
    return is_jax_enabled