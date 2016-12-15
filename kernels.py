from MPLAnimator.MPLAnimator import Animator
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

matplotlib.use("Qt5Agg")


N = 81


def kernel(phi, x, x_, args):
    if len(np.asarray(args).shape) == 1:
        args = [(arg,) for arg in args]
    return np.sum(phi(x, *arg) * phi(x_, *arg) for arg in args)


def polynomial(x, degree):
    return x**degree


def sigmoid(x, loc=0, k=1):
    return 1 / (1 + np.exp(-k*(x-loc)))


def setup():
    plt.gcf().set_size_inches(12, 6)
    plt.suptitle("From Basis Functions φi(x) to Kernels k(x,x') = φ(x)T φ(x)")


def plt_poly(i):
    x = np.linspace(-1, 1, N)
    degrees = range(1, 10)

    ax = plt.subplot(231)
    ax.cla()
    ax.set_title('Polynomial basis functions M = {}'.format(len(degrees)))
    for degree in degrees:
        y = polynomial(x, degree)
        ax.plot(x, y)
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)

    ax = plt.subplot(234)
    ax.cla()
    ax.set_title('Polynomial kernel: x\' = {0:.2f}'.format(x[i]))
    y = kernel(polynomial, x, x[i], degrees)
    ax.plot(x, y)
    ax.plot(x[i], -1, 'xr')
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)


def plt_gaus(i):
    x = np.linspace(-1, 1, N)
    locs = list(np.linspace(-1, 1, 11))
    scales = np.repeat(.4, len(locs))

    ax = plt.subplot(232)
    ax.cla()
    ax.set_title('Gaussian basis functoions M = {}'.format(len(locs)))
    for j, loc in enumerate(locs):
        ax.plot(x, norm.pdf(x, loc, scales[j]))
        ax.set_ylim(0, 1)
        ax.set_xlim(-1, 1)

    ax = plt.subplot(235)
    ax.cla()
    ax.set_title('Gaussian kernel: x\' = {0:.2f}'.format(x[i]))
    y = kernel(norm.pdf, x, x[i], zip(locs, scales))
    ax.plot(x, y)
    ax.plot(x[i], 0, 'xr')
    ax.set_ylim(0, 4)
    ax.set_xlim(-1, 1)


def plt_sigm(i):
    x = np.linspace(-1, 1, N)
    locs = np.linspace(-1.1, 1.1, 11)
    scales = np.repeat(10, len(locs))

    ax = plt.subplot(233)
    ax.cla()
    ax.set_title('Sigmoidal basis functoions M = {}'.format(len(locs)))
    for j, loc in enumerate(locs):
        y = sigmoid(x, loc, scales[j])
        ax.plot(x, y)
        ax.set_ylim(0, 1)
        ax.set_xlim(-1, 1)

    ax = plt.subplot(236)
    ax.cla()
    ax.set_title("Sigmoid kernel: x\' = {0:.2f}".format(x[i]))
    y = kernel(sigmoid, x, x[i], zip(locs, scales))
    ax.plot(x, y)
    ax.plot(x[i], 0, 'xr')
    ax.set_ylim(0, 6)
    ax.set_xlim(-1, 1)


def frame(i):
    plt_poly(i)
    plt_gaus(i)
    plt_sigm(i)


a = Animator(name='Kernels', setup_handle=setup)
a.setFrameCallback(frame_handle=frame, max_frame=N)
a.run(clear=False, precompile=True, initialFrame=N//2)
