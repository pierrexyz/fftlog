is_jax_enabled = False  # Default setting

def set_jax_enabled(value: bool):
    global is_jax_enabled
    is_jax_enabled = value

def get_jax_enabled():
    return is_jax_enabled