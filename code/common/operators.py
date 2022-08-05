import numpy as np
import pywt
import pywt.data
from random import randint


##########################################################################
# Operators for Exercise 1
##########################################################################


def l1_prox(y, weight):
    """Projection onto the l1-ball.
    """
    return np.sign(y)*np.maximum(np.absolute(y) - weight, 0)


def l2_prox(y, weight):
    """Projection onto the l2-ball.
    """
    return (1.0 / (weight + 1)) * y


def norm1(x):
    """Returns the l1 norm `x`.
    """
    return np.linalg.norm(x, 1)


def norm2sq(x):
    """Returns the l2 norm squared of `x`.
    """
    return (1.0 / 2) * np.linalg.norm(x) ** 2


def stocgradfx(x, minibatch_size, A, b):
    max_rand = A.shape[0]
    grad_est = np.zeros(x.shape)
    for i in range(0, minibatch_size):
        index = randint(0, max_rand - 1)
        grad_est += stocgradfx_single(x, index, A, b)

    return grad_est / minibatch_size

def stocgradfx_single(x, i, A, b):
    di = np.exp(-np.dot(b[i], np.dot(A[i],x)))
    gradfx = (-b[i] * A[i] * di / (1+di)).T
    return gradfx


def gradfx(x, A, b):
    n = A.shape[0]
    gradfx = np.zeros(x.shape)
    for i in range(n):
        gradfx += stocgradfx_single(x,i,A,b)
    gradfx = gradfx/n

    return gradfx


def fx(x, A, b):
    fx = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        ci = b[i]*np.dot(A[i],x)
        fx[i] = np.log(1+np.exp(-ci))
    fx = fx.mean()
    return fx


##########################################################################
# Operators for Exercise 2
##########################################################################

def TV_norm(X, opt=None):
    """
        Computes the TV-norm of image X
        opts = 'iso' for isotropic, else it is the anisotropic TV-norm
    """

    m, n = X.shape
    P1 = X[0:m - 1, :] - X[1:m, :]
    P2 = X[:, 0:n - 1] - X[:, 1:n]

    if opt == 'iso':
        D = np.zeros_like(X)
        D[0:m - 1, :] = P1 ** 2
        D[:, 0:n - 1] = D[:, 0:n - 1] + P2 ** 2
        tv_out = np.sum(np.sqrt(D))
    else:
        tv_out = np.sum(np.abs(P1)) + np.sum(np.abs(P2))

    return tv_out


# P_Omega and P_Omega_T
def p_omega(x, indices):  # P_Omega

    return np.expand_dims(x[np.unravel_index(indices, x.shape)], 1)


def p_omega_t(x, indices, m):  # P_Omega^T
    y = np.zeros((m, m))
    y[np.unravel_index(indices, y.shape)] = x.squeeze()
    return y


class RepresentationOperator(object):
    """
        Representation Operator contains the forward and adjoint
        operators for the Wavelet transform.
    """

    def __init__(self, m=256):
        self.m = m
        self.N = m ** 2

        self.W_operator = lambda x: pywt.wavedec2(x, 'db8', mode='periodization')  # From image coefficients to wavelet
        self.WT_operator = lambda x: pywt.waverec2(x, 'db8', mode='periodization')  # From wavelet coefficients to image
        _, self.coeffs = pywt.coeffs_to_array(self.W_operator(np.ones((m, m))))

    def W(self, x):
        """
            Computes the Wavelet transform from a vectorized image.
        """
        x = np.reshape(x, (self.m, self.m))
        wav_x, _ = pywt.coeffs_to_array(self.W_operator(x))

        return np.reshape(wav_x, (self.N, 1))

    def WT(self, wav_x):
        """
            Computes the adjoint Wavelet transform from a vectorized image.
        """
        wav_x = np.reshape(wav_x, (self.m, self.m))
        x = self.WT_operator(pywt.array_to_coeffs(wav_x, self.coeffs, output_format='wavedec2'))
        return np.reshape(x, (self.N, 1))


