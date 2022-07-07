from __future__ import generators
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class zono:

    def __init__(self, values: np.array = None, dimension = 1, generators = 1) -> None:
        if values is None:
            self.values = np.zeros((dimension, generators))
        else:
            self.values = values
        self.dimensions = self.values.shape[0]
        self.generators = self.values.shape[1] - 1
    
    def __str__(self) -> str:
        return str(self.values)
    
    def __repr__(self) -> str:
        return str(self.values)
    
    def __add__(self, other: 'zono') -> 'zono':
        """
        Add two zonotopes with the same dimension. 
        If the dimension is different, an value error is raised.
        The zonootopes don't have to have the same number of generators. 
        If they don't, the generators are padded with zeros.
        
        Args:
            other: zonotope to add.
        
        Returns:
            zonotope sum.
        """
        if self.values.shape[0] != other.values.shape[0]:
            raise ValueError("Dimension mismatch")
        if self.values.shape[1] == other.values.shape[1]:
            return zono(np.add(self.values, other.values))
        if self.values.shape[1] > other.values.shape[1]:
            other = np.pad(other.values, [(0, 0), (0, self.values.shape[1] - other.values.shape[1])], 'constant', constant_values = 0)
            return zono(np.add(self.values, other))
        else:
            v = np.pad(self.values, [(0, 0), (0, other.values.shape[1] - self.values.shape[1])], 'constant', constant_values = 0)
            return zono(np.add(v, other.values))
    
    def __mul__(self, other: float) -> 'zono':
        return zono(np.multiply(self.values, other))
    
    def __mul__(self, other: int) -> 'zono':
        return zono(np.multiply(self.values, other))
    
    def __rmul__(self, other: float) -> 'zono':
        return self.__mul__(other)
    
    def __rmul__(self, other: int) -> 'zono':
        return self.__mul__(other)
    
    def combine(self, other: 'zono') -> 'zono':
        return zono(values = np.append(self.values, other.values, axis=0))
    
    def split(self) -> 'zono':
        if self.dimensions % 2 != 0:
            raise ValueError("Dimension must be even")
        else:
            values = np.split(self.values, 2, axis=0)
            return zono(values[0]), zono(values[1])
    
    def upper_bound(self, dimension = 1) -> float:
        bound = self.values[dimension-1][0]
        for g in self.values[dimension-1][1:]:
            if g > 0:
                bound += g
            else:
                bound -= g
        return bound
    
    def lower_bound(self, dimension = 1) -> float:
        bound = self.values[dimension-1][0]
        for g in self.values[dimension-1][1:]:
            if g > 0:
                bound -= g
            else:
                bound += g
        return bound
    
    def visualize(self, quiver = False, shape = False, fig = None, ax = None) -> None:
        show_self = (fig == None) or (ax == None)
        if self.dimensions > 2:
            raise ValueError("Dimension must be <= 2")
        else:
            if fig is None:
                fig = plt.figure()
            if ax is None:
                ax = fig.add_subplot(111, aspect = 'equal')
                ax.set_xlim(0, 4)
                ax.set_ylim(0, 5)
            if self.dimensions == 1:
                pass #TODO implement 1D visualization
            else:
                if quiver:
                    for i in range(self.generators):
                        ax.quiver(self.values[0][0], self.values[1][0], self.values[0][i + 1], self.values[1][i + 1])
                        ax.quiver(self.values[0][0], self.values[1][0], -self.values[0][i + 1], -self.values[1][i + 1])
                if shape and self.generators > 1:
                    x = []
                    for i in itertools.product([-1.0, 1.0], repeat = self.generators):
                        x.append((np.sum(self.values[0][1:] * np.array(list(i))) + self.values[0][0]))
                    y = []
                    for i in itertools.product([-1.0, 1.0], repeat = self.generators):
                        y.append((np.sum(self.values[1][1:] * np.array(list(i))) + self.values[1][0]))
                    print(x)
                    x, y = np.array(x), np.array(y)
                    order = np.argsort(np.arctan2(y - y.mean(), x - x.mean()))
                    ax.fill(x[order], y[order], "g", alpha=0.5)
            if show_self:
                plt.show()


if __name__ == "__main__":
    input = zono(values = np.array([[2, 1, 0], [3, 1, 1]]))
    print(f"input: {input}")
    print(input.generators)
    input.visualize(shape=True, quiver=True)

        

    

