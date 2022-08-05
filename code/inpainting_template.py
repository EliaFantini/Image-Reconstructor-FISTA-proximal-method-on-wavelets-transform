import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import structural_similarity as ssim

from common.utils import apply_random_mask, psnr, load_image, print_progress, print_end_message, print_start_message
from common.operators import TV_norm, RepresentationOperator, p_omega, p_omega_t, l1_prox, norm1, norm2sq



def ISTA(fx, gx, gradf, proxg, params, verbose = False):
    method_name = 'ISTA'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize parameters.
    x0 = params['x0']
    maxit = params['maxit']
    lmbd = params['lambda']
    alpha = 1 / params['prox_Lips']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update the iterate
        y = x_k - alpha * gradf(x_k)
        x_k_next = proxg(y, alpha * lmbd)
        x_k = x_k_next

        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % params['iter_print'] == 0 and verbose:
            print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

    if verbose:
        print_end_message(method_name, time.time() - tic_start)
    return x_k, info


def FISTA(fx, gx, gradf, proxg, params, verbose=False):
    '''
    if params['restart_fista']:
        method_name = 'FISTAR'
    else:
        method_name = 'FISTA'
        '''
    method_name = 'FISTA' # added
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize parameters
    x0 = params['x0']
    maxit = params['maxit']
    lmbd = params['lambda']
    alpha = 1 / params['prox_Lips']
    y_k = x0
    t_k = 1
    restart_fista = params['restart_criterion']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update iterate
        prox_argument = y_k - alpha * gradf(y_k)
        x_k_next = proxg(prox_argument, alpha)
        t_k_next = (1 + np.sqrt(4 * (t_k ** 2) + 1)) / 2
        y_k_next = x_k_next + ((t_k - 1) / t_k_next) * (x_k_next - x_k)
        if restart_fista and gradient_scheme_restart_condition(x_k.reshape(x_k.shape[0],), x_k_next.reshape(x_k_next.shape[0],), y_k.reshape(y_k.shape[0],)):
            y_k = x_k
        else:
            y_k = y_k_next
            t_k = t_k_next
            x_k = x_k_next


        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % params['iter_print'] == 0:
            if verbose:
                print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

    if verbose:
        print_end_message(method_name, time.time() - tic_start)
    return x_k, info


def gradient_scheme_restart_condition(x_k, x_k_next, y_k):
    """
    Whether to restart
    """
    return (y_k - x_k_next) @ (x_k_next - x_k) > 0

def reconstructL1(image, indices, optimizer, params):
    # Wavelet operator
    r = RepresentationOperator(m=params["m"])

    # Define the overall operator
    forward_operator = lambda x: p_omega(r.WT(x), indices)  # P_Omega.W^T
    adjoint_operator = lambda x: r.W(p_omega_t(x,indices,params['m']))  # W. P_Omega^T

    # Generate measurements
    b = p_omega(image, indices)

    fx = lambda x: norm2sq(b - forward_operator(x))
    gx = lambda x:  norm1(x)
    proxg = lambda x, y: l1_prox(x, params['lambda'] * y)
    gradf = lambda x: adjoint_operator(forward_operator(x) - b)

    x, info = optimizer(fx, gx, gradf, proxg, params, verbose=params['verbose'])
    return r.WT(x).reshape((params['m'], params['m'])), info


def reconstructTV(image, indices, optimizer, params):
    """
        image: undersampled image (mxm) to be reconstructed
        indices: indices of the undersampled locations
        optimizer: method of reconstruction (FISTA/ISTA function handle)
        params:
    """
    # Define the overall operator
    forward_operator = lambda x: p_omega(x,indices)  # P_Omega
    adjoint_operator = lambda x: p_omega_t(x,indices, params['m']) # P_Omega^T

    # Generate measurements
    b = forward_operator(image)

    fx = lambda x: norm2sq(b - forward_operator(x))
    gx = lambda x: TV_norm(x, optimizer)
    proxg = lambda x, y: denoise_tv_chambolle(x.reshape((params['m'], params['m'])),
                                              weight=params["lambda"] * y, eps=1e-5,
                                              n_iter_max=50).reshape((params['N'], 1))
    gradf = lambda x: adjoint_operator(forward_operator(x) - b).reshape(x.shape[0],1)

    x, info = optimizer(fx, gx, gradf, proxg, params, verbose=params['verbose'])
    return x.reshape((params['m'], params['m'])), info


# %%

if __name__ == "__main__":

    ##############################
    # Load image and sample mask #
    ##############################
    shape = (256, 256)
    params = {
        'maxit': 200,
        'tol': 10e-15,
        'prox_Lips': 1,
        'lambda': 0.01,
        'x0': np.zeros((shape[0] * shape[1], 1)),
        'restart_criterion': True,
        'stopping_criterion': False,
        'iter_print': 50,
        'shape': shape,
        'restart_param': 50,
        'verbose': True,
        'm': shape[0],
        'rate': 0.4,
        'N': shape[0] * shape[1]
    }
    PATH = 'data/gandalf.jpg'
    image = load_image(PATH, params['shape'])


    im_us, mask = apply_random_mask(image, params['rate'])
    indices = np.nonzero(mask.flatten())[0]
    params['indices'] = indices
    # Choose optimization parameters


    lambdas = np.logspace(-4, 0, 10)
    psnr_l1_list = []
    psnr_tv_list = []

    #paramater sweep over lambda, getting the one with better psnr score
    for lambda_ in lambdas:
        params['lambda'] = lambda_
        #######################################
        # Reconstruction with L1 and TV norms #
        #######################################
        t_start = time.time()
        reconstruction_l1 = reconstructL1(image, indices, FISTA, params)[0]
        t_l1 = time.time() - t_start

        psnr_l1 = psnr(image, reconstruction_l1)
        ssim_l1 = ssim(image, reconstruction_l1)
        psnr_l1_list.append(psnr_l1)

        t_start = time.time()
        reconstruction_tv = reconstructTV(image, indices, FISTA, params)[0]
        t_tv = time.time() - t_start

        psnr_tv = psnr(image, reconstruction_tv)
        ssim_tv = ssim(image, reconstruction_tv)
        psnr_tv_list.append(psnr_tv)
    #using the lambda with maximum psnr score to reconstruct the image
    max_psnr_l1 = max(psnr_l1_list)
    best_lambda_l1 = lambdas[psnr_l1_list.index(max_psnr_l1)]
    max_psnr_tv = max(psnr_tv_list)
    best_lambda_tv = lambdas[psnr_tv_list.index(max_psnr_tv)]

    t_start = time.time()
    params['lambda']= best_lambda_l1
    reconstruction_l1 = reconstructL1(image, indices, FISTA, params)[0]
    t_l1 = time.time() - t_start

    psnr_l1 = psnr(image, reconstruction_l1)
    ssim_l1 = ssim(image, reconstruction_l1)

    t_start = time.time()
    params['lambda'] = best_lambda_tv
    reconstruction_tv = reconstructTV(image, indices, FISTA, params)[0]
    t_tv = time.time() - t_start

    psnr_tv = psnr(image, reconstruction_tv)
    ssim_tv = ssim(image, reconstruction_tv)

    # Plot the reconstructed image alongside the original image and PSNR
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0][0].imshow(image, cmap='gray')
    ax[0][0].set_title('Original')
    ax[0][1].imshow(im_us, cmap='gray')
    ax[0][1].set_title('Original with missing pixels')
    ax[1][0].imshow(reconstruction_l1, cmap="gray")
    ax[1][0].set_title('L1 - PSNR = {:.2f}\n SSIM  = {:.2f} - Time: {:.2f}s'.format(psnr_l1, ssim_l1, t_l1))
    ax[1][1].imshow(reconstruction_tv, cmap="gray")
    ax[1][1].set_title('TV - PSNR = {:.2f}\n SSIM  = {:.2f}  - Time: {:.2f}s'.format(psnr_tv, ssim_tv, t_tv))
    [axi.set_axis_off() for axi in ax.flatten()]
    plt.tight_layout()
    plt.show()
    #plotting how psnr score changes with the changing of lambda parameter
    plt.semilogx(lambdas, psnr_l1_list, label= 'l1-norm')
    plt.semilogx(lambdas, psnr_tv_list, label= 'tv-norm')
    plt.xlabel("Lambda")
    plt.ylabel("Psnr")
    plt.grid()
    plt.show()
