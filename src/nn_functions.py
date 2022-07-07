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

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 3)

    ax[0].set_title("Input")
    ax[0].set_xlim(-1, 4.5)
    ax[0].set_ylim(0, 4)
    ax[1].set_title("Hidden")
    ax[1].set_xlim(-1, 4.5)
    ax[1].set_ylim(0, 4)
    ax[2].set_title("Output")
    ax[2].set_xlim(-1, 4.5)
    ax[2].set_ylim(0, 4)

    input = zono(values = np.array([[1, 1, 0], [2, 0, 1]]))
    print(f"input: {input}")
    input.visualize(shape=True, fig=fig, ax=ax[0])
    i1, i2 = input.split()
    v1 = relu(affine([1, 0.7], i1, i2))
    v2 = relu(affine([0.2, 0.45], i1, i2))
    hidden = v1.combine(v2)
    print(f"hidden: {hidden}")
    hidden.visualize(shape=True, fig=fig, ax=ax[1])
    o1 = relu(affine([0.3, 0.1], v1, v2))
    o2 = relu(affine([0.1, 0.2], v1, v2))
    print(f"o1: {o1}")
    print(f"o2: {o2}")
    output = o1.combine(o2)
    print(f"output: {output}")
    output.visualize(shape=True, fig=fig, ax=ax[2])

    i1 = (0, 2)
    i2 = (1, 3)
    interval.visualize(i1, i2, fig=fig, ax=ax[0])
    v1 = interval.relu(interval.affine([1, 0.7], i1, i2))
    v2 = interval.relu(interval.affine([0.2, 0.45], i1, i2))
    interval.visualize(v1, v2, fig=fig, ax=ax[1])
    o1 = interval.relu(interval.affine([0.3, 0.1], v1, v2))
    o2 = interval.relu(interval.affine([0.1, 0.2], v1, v2))
    interval.visualize(o1, o2, fig=fig, ax=ax[2])

    plt.show()
