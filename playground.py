import numpy as np
from numpy.random import normal, random_sample, permutation
from numpy.linalg import norm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

orangeblue = LinearSegmentedColormap('OrangeBlue', {
    'red':   ((0.0,  1.0, 1.0),
              (0.5,  1.0, 1.0),
              (1.0,  0.0, 0.0)),
    'green': ((0.0,  0.5, 0.5),
              (0.5,  1.0, 1.0),
              (1.0,  0.0, 0.0)),
    'blue':  ((0.0,  0.0, 0.0),
              (0.5,  1.0, 1.0),
              (1.0,  1.0, 1.0))
})


def viz(data, labels, progress, accuracy, fig, discretize=False):
    """Function visualizing training progress."""
    fig.suptitle('{:3.2%} done with accuracy {:2.4f}'.format(progress, accuracy))
    labels = np.round(labels) if discretize else labels
    plt.scatter(*zip(*data), c=labels, cmap=orangeblue)
    fig.canvas.draw()


class DataGenerator:
    """Provides dataset generators and helper functions."""

    # Returns n floats uniformly distributed between a and b.
    uniform = lambda self, a, b, n=1: (b - a) * random_sample(n) + a

    def split(self, sets, split=10):
        """Splits samples into train and test sets and reshapes target labels as required."""
        n = sets[0].shape[0]
        permut = permutation(n).astype(int)
        return tuple(zip(*((d[permut[n//split:]], d[permut[:n//split]]) for d in sets)))

    def circle(self, n=1000, radius=5, noise=0):
        """Generates data with one class describing an inner and one an outer circle."""
        r = np.append(self.uniform(0, radius*.5, n//2),
                      self.uniform(radius*.55, radius, n//2))
        angle = self.uniform(0, 2*np.pi, n)
        x = r * np.sin(angle) + self.uniform(-radius, radius, n) * noise
        y = r * np.cos(angle) + self.uniform(-radius, radius, n) * noise
        t = np.less(norm(np.vstack((x, y)), axis=0), radius*.5)
        return x, y, t

    def xor(self, n=1000, padding=.3, noise=0):
        """Generates xor data."""
        x = self.uniform(-5, 5, n); x[x > 0] += padding; x[x < 0] -= padding
        y = self.uniform(-5, 5, n); y[y > 0] += padding; y[y < 0] -= padding
        x += self.uniform(-5, 5, n) * noise
        y += self.uniform(-5, 5, n) * noise
        t = np.less(x*y, 0).flatten()
        return x, y, t

    def gauss(self, n=1000, noise=0):
        """Generates two blops of gauss data, each describing a class."""
        x = normal( 2, 1+noise, (n//2, 2))
        y = normal(-2, 1+noise, (n//2, 2))
        t = np.append(np.ones(n//2), np.zeros(n//2))
        return (np.vstack((x, y)), t)

    def genSpiral(self, delta, n, noise=0):
        """Generates a spiral of n points with rotation-offset delta."""
        points = np.arange(n)
        r = points / n * 5
        t = 1.75 * points / n * 2 * np.pi + delta
        noise = self.uniform(-1, 1, (2,n)) * noise
        xy = np.vstack((r * np.sin(t), r * np.cos(t))) + noise
        return xy.T


    def spiral(self, n=1000, noise=0):
        """Generates a dataset of two intertwined spirals."""
        a = self.genSpiral(    0, n//2, noise)
        b = self.genSpiral(np.pi, n//2, noise)
        t = np.append(np.ones(n//2), np.zeros(n//2))
        return (np.vstack((a, b)), t)
