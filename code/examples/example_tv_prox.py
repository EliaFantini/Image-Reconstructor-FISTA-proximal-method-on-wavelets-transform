import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import *
from common.operators import TV_norm

def main():
    m = 256

    # Set TV prox parameters
    maxit = 100
    tol = 1e-5
    reg_param = 1

    # Compute TV norm of random image and denoise using the prox of TV norm
    I = np.random.randn(m, m)
    I_tv_denoised = denoise_tv_chambolle(I, weight=reg_param, eps=tol, n_iter_max=maxit)

    fig, ax = plt.subplots(1, 2, figsize=(8, 5))

    ax[0].imshow(I)
    ax[0].set_title('TV norm = {:.2f}'.format(TV_norm(I)))
    ax[1].imshow(I_tv_denoised)
    ax[1].set_title('TV norm = {:.2f}'.format(TV_norm(I_tv_denoised)))
    plt.show()


if __name__ == "__main__":
    main()
