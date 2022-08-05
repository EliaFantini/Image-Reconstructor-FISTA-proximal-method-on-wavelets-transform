import time

import os
from numpy import diag, exp, sum, equal, argmax, where
from numpy.linalg import norm
import pickle
import common.operators as operators
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import PIL
from PIL import Image
import torchvision.transforms.functional as tf

DATA_PATH = './data'
RESULTS_PATH = './results'
RESULTS_FILE = './results/results.pickle'
DATA_FILE = './data/mnist_data.pickle'
MNIST_TRAIN_TEST_SPLIT = './data/mnist_train_test_split.npz'
ACCESS_RIGHTS = 0o777


def load_mnist():
    data = np.load(MNIST_TRAIN_TEST_SPLIT)

    print('\n----- MNIST datasets loaded: {:d} training samples and {:d} test samples.'.format(data['A_train'].shape[0], data['A_test'].shape[0]))
    return data['A_train'], data['b_train'], data['b_train_binarized'],\
           data['A_test'], data['b_test'], 10, data['A_train'].shape[1]


def get_operators_based_on_data(A_train, b_train, b_train_binarized):
    fx = lambda Y: operators.fx(Y, A_train, b_train)
    gradfx = lambda Y: operators.gradfx(Y, A_train, b_train_binarized)
    stocgradfx = lambda Y, index: operators.stocgradfx(Y, index, A_train, b_train_binarized)
    Lips = (1 / 2) * norm(A_train, 'fro') ** 2
    Lips_max = (1 / 2) * get_max_l2_row_norm(A_train)

    return fx, gradfx, stocgradfx, Lips, Lips_max


def compute_accuracy(X_hat, A_test, b_test):
    num_test_samples = len(b_test)
    denominators = 1.0 / sum(exp(A_test @ X_hat), axis=1)
    Z = diag(denominators)
    predictions = argmax(Z @ exp(A_test @ X_hat), axis=1)
    correct = len(where(equal(b_test, predictions))[0])

    return correct / num_test_samples


def print_end_message(method_name, time_spent):
    print('\n---------- Training over - {:s}. Took {:d} seconds. \n\n'.format(method_name, np.math.ceil(time_spent)))


def print_start_message(method_name):
    print('\n\n\n---------- Optimization with {:s} started.\n\n'.format(method_name))


def print_progress(i, maxit, val_F, val_f, val_g):
    print('\n--- iter = {:d}/{:d}, F(X) = {:f}, f(X) = {:f}, g(X) = {:f}.'.format(i, maxit, val_F, val_f, val_g))

def read_f_star(fx, lmbd_l1, lmbd_l2):
    with open('data/argmin_l1_reg.pickle', 'rb') as f:
        X_opt_l1 = pickle.load(f)
    f_star_l1 = fx(X_opt_l1) + lmbd_l1 * operators.norm1(X_opt_l1)

    with open('data/argmin_l2_reg.pickle', 'rb') as f:
        X_opt_l2 = pickle.load(f)
    f_star_l2 = fx(X_opt_l2) + lmbd_l2 * operators.norm2sq(X_opt_l2)

    return f_star_l1, X_opt_l1, f_star_l2, X_opt_l2

def get_max_l2_row_norm(A):
    n = A.shape[0]
    print(n)
    time.sleep(3)
    maxrnorm_sq = -1
    for i in range(0, n):
        curr_norm = np.linalg.norm(A[i, :], 2)
        if maxrnorm_sq < curr_norm:
            maxrnorm_sq = curr_norm

    assert (maxrnorm_sq != -1)

    return n * maxrnorm_sq


def apply_random_mask(image, rate=0.25):
    """
        apply_random_mask takes an image and applies a random undersampling mask at the given rate.
        It returns the undersampled image with the corresponding mask.
    """
    if isinstance(image, np.ndarray):
        mask = np.random.random(image.shape)
    else:
        mask = torch.rand(image.shape)
    mask[mask > rate] = 0
    mask[mask > 0.] = 1
    return image * mask, mask


def psnr(ref, recon):
    """
        psnr takes as input the reference (ref) and the estimated (recon) images.
    """
    mse = np.sqrt(((ref - recon) ** 2).sum() / np.prod(ref.shape))
    return 20 * np.log10(ref.max() / mse)


def load_image(path, size=(256, 256)):
    """
    load_image loads the image at the given path and resizes it to the given size
    with bicubic interpolation before normalizing it to 1.
    """
    I = Image.open(path)
    I = tf.resize(I, size, interpolation=PIL.Image.BICUBIC)
    I = tf.to_grayscale(I)
    I = tf.to_tensor(I).numpy()[0, :, :]
    return I
