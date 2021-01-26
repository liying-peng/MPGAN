import skimage.metrics as measure
import numpy as np
import itk


def compute_psnr(ground_truth, test_image):
    '''
    :param ground_truth: the ground truth images, format: [batch_size, w, h, d, c]
    :param test_image: the image you want to test
    :return: PSNR for a batch
    '''
    psnr = 0.0

    for i, item in enumerate(test_image):
        psnr_per_image = measure.peak_signal_noise_ratio(ground_truth[i, :, :, :, 0], item[:, :, :, 0])
        psnr += psnr_per_image

    return float(psnr / (i + 1))


def compute_ssim(ground_truth, test_image):
    '''
    :param ground_truth: the ground truth images, format: [batch_size, w, h, d, c]
    :param test_image: the image you want to test
    :return: SSIM for a batch
    '''
    ssim = 0.0

    for i, item in enumerate(test_image):
        ssim_per_image = measure.structural_similarity(ground_truth[i, :, :, :, 0], item[:, :, :, 0])
        ssim += ssim_per_image

    return float(ssim / (i + 1))


def compute_pearson_correlation(a, b):
    a_m = a - np.mean(a)
    b_m = b - np.mean(b)
    inner_product = np.sum(a_m * b_m)
    a_norm = np.sqrt(np.sum(a_m * a_m))
    b_norm = np.sqrt(np.sum(b_m * b_m))

    return inner_product / (a_norm * b_norm)
