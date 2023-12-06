import tensorflow.compat.v1 as tf
import numpy as np


def uniform(shape, scale=0.05, name=None):
    """Initialize weights with values sampled from a uniform distribution.

    Args:
        shape (tuple): The shape of the weight tensor.
        scale (float): Scaling factor for the uniform distribution (default is 0.05).
        name (str): Name for the variable (optional).

    Returns:
        tf.Variable: Variable initialized with values sampled from a uniform distribution.
    """
    
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
     """Initialize weights using Glorot initialization (also known as Xavier initialization).

    Glorot initialization is designed to keep the scale of the gradients roughly constant.

    Args:
        shape (tuple): The shape of the weight tensor.
        name (str): Name for the variable (optional).

    Returns:
        tf.Variable: Variable initialized with values based on Glorot initialization.
    """
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
     """Initialize weights with zeros.

    Args:
        shape (tuple): The shape of the weight tensor.
        name (str): Name for the variable (optional).

    Returns:
        tf.Variable: Variable initialized with zeros.
    """
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
     """Initialize weights with ones.

    Args:
        shape (tuple): The shape of the weight tensor.
        name (str): Name for the variable (optional).

    Returns:
        tf.Variable: Variable initialized with ones.
    """
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
