import numpy as np
import interval_abstraction as interval

from zonotope import zono
from matplotlib import pyplot as plt

def relu(zono_input: zono) -> zono:
    """
    Compute the ReLU function on a 1 dimensional zonotope.

    Args:
        zono_input: one dimensional zonotope input.
    
    Returns:
        one dimensional zonotope output.
    """
    if zono_input.upper_bound() < 0:
        return zono(dimension = zono_input.dimensions, generators = zono_input.generators)
    elif zono_input.lower_bound() > 0:
        return zono_input
    else:
        la = zono_input.upper_bound() / (zono_input.upper_bound()- zono_input.lower_bound())
        em = (zono_input.upper_bound() * (1 - la)) / 2
        zono_output = la * zono(values = (np.pad(zono_input.values, [(0, 0), (0, 1)], 'constant', constant_values = 0)))
        new_zono = zono(dimension = zono_output.dimensions, generators = zono_output.generators)
        new_zono.values[0][0] = em
        new_zono.values[0][-1] = em
        return zono_output + new_zono

def affine(weights, *zono_input : zono) -> zono:
    """
    Compute the affine function on a zonotope.

    Args:
        weights: weights of the affine function.
        zono_input: zonotope inputs.
    
    Returns:
        zonotope output.
    """
    if len(zono_input) != len(weights):
        raise ValueError("Dimension mismatch")
    zono_output = zono(dimension = zono_input[0].dimensions, generators = zono_input[0].generators)
    for i in range(len(zono_input)):
        zono_output += weights[i] * zono_input[i]
    return zono_output
