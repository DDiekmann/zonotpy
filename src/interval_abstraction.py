import matplotlib.pyplot as plt
import matplotlib.patches as patches

def relu(interval):
    l = 0
    u = 0
    if interval[0] > 0:
        l = interval[0]
    if interval[1] > 0:
        u = interval[1]
    return (l, u)

def affine(weights, *intervals):
    if len(intervals) != len(weights):
        raise ValueError("Dimension mismatch")
    l = 0
    u = 0
    for i in range(len(intervals)):
        l += weights[i] * intervals[i][0]
        u += weights[i] * intervals[i][1]
    return (l, u)

def visualize(interval_x, interval_y, fig, ax):
    ax.add_patch(patches.Rectangle((interval_x[0], interval_y[0]), interval_x[1] - interval_x[0], interval_y[1] - interval_y[0], fill=False))
