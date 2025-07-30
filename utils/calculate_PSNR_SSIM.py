
import math
import numpy as np
import cv2
from torchvision.utils import make_grid


####################
# image convert
####################


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)



####################
# metric
####################


def calculate_psnr(origin, pred):
    # origin and pred have range [0, 255]
    origin = origin.astype(np.float64)
    pred = pred.astype(np.float64)
    mse = np.mean((origin - pred)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse)), mse


def calculate_apd(origin, pred):
    origin = origin.astype(np.float64)
    pred = pred.astype(np.float64)

    abs_diff = np.abs(origin - pred)
    apd = np.mean(abs_diff)

    return apd


def ssim(origin, pred):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    origin = origin.astype(np.float64)
    pred = pred.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(origin, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(pred, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(origin**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(pred**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(origin * pred, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(origin, pred):
    '''calculate SSIM
    the same outputs as MATLAB's
    origin, pred: [0, 255]
    '''
    if not origin.shape == pred.shape:
        raise ValueError('Input images must have the same dimensions.')
    if origin.ndim == 2:
        return ssim(origin, pred)
    elif origin.ndim == 3:
        if origin.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(origin, pred))
            return np.array(ssims).mean()
        elif origin.shape[2] == 1:
            return ssim(np.squeeze(origin), np.squeeze(pred))
    else:
        raise ValueError('Wrong input image dimensions.')


def tensor_psnr_ssim(origin, pred):

    origin = origin.cpu().detach()
    pred = pred.cpu().detach()
    origin_array = tensor2img(origin)
    pred_array = tensor2img(pred)

    apd = calculate_apd(origin_array, pred_array)
    psnr, mse = calculate_psnr(origin_array, pred_array)
    ssim = calculate_ssim(origin_array, pred_array)

    return apd, mse, psnr, ssim
