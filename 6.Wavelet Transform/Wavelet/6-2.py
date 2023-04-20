import matplotlib.pyplot as plt
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from skimage.io import imread

original = img_as_float(imread('Lena.bmp'))

sigma = 0.2
noisy = random_noise(original, var=sigma ** 2)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 5),
                       sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, average_sigmas=True, multichannel=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
print(f"Estimated Gaussian noise standard deviation = {sigma_est}")

im_bayes_soft = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True, wavelet='haar',
                                method='BayesShrink', mode='soft',
                                rescale_sigma=True)
im_bayes_hard = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True, wavelet='haar',
                                method='BayesShrink', mode='hard',
                                rescale_sigma=True)
im_visushrink_soft = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True, wavelet='haar',
                                     method='VisuShrink', mode='soft',
                                     sigma=sigma_est, rescale_sigma=True)
im_visushrink_hard = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True, wavelet='haar',
                                method='VisuShrink', mode='hard',
                                sigma=sigma_est, rescale_sigma=True)

# VisuShrink is designed to eliminate noise with high probability, but this
# results in a visually over-smooth appearance.  Repeat, specifying a reduction
# in the threshold by factors of 2 and 4.
im_visushrink2 = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True, wavelet='haar',
                                 method='VisuShrink', mode='soft',
                                 sigma=sigma_est / 2, rescale_sigma=True)
im_visushrink4 = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True, wavelet='haar',
                                 method='VisuShrink', mode='soft',
                                 sigma=sigma_est / 4, rescale_sigma=True)

# Compute PSNR as an indication of image quality
psnr_noisy = peak_signal_noise_ratio(original, noisy)
psnr_bayes_soft = peak_signal_noise_ratio(original, im_bayes_soft)
psnr_bayes_hard = peak_signal_noise_ratio(original, im_bayes_hard)
psnr_visushrink_soft = peak_signal_noise_ratio(original, im_visushrink_soft)
psnr_visushrink_hard = peak_signal_noise_ratio(original, im_visushrink_hard)
psnr_visushrink2 = peak_signal_noise_ratio(original, im_visushrink2)
psnr_visushrink4 = peak_signal_noise_ratio(original, im_visushrink4)

ax[0, 0].imshow(noisy)
ax[0, 0].axis('off')
ax[0, 0].set_title('Noisy\nPSNR={:0.4g}'.format(psnr_noisy))
ax[0, 1].imshow(im_bayes_soft)
ax[0, 1].axis('off')
ax[0, 1].set_title(
    'Wavelet denoising\n(BayesShrink Soft)\nPSNR={:0.4g}'.format(psnr_bayes_soft))

ax[0, 2].imshow(im_bayes_hard)
ax[0, 2].axis('off')
ax[0, 2].set_title(
    'Wavelet denoising\n(BayesShrink Hard)\nPSNR={:0.4g}'.format(psnr_bayes_hard))
ax[1, 0].imshow(original)
ax[1, 0].axis('off')
ax[1, 0].set_title('Original')
ax[1, 1].imshow(im_visushrink2)
ax[1, 1].axis('off')
ax[1, 1].set_title(
    'Wavelet denoising\n(VisuShrink, $\\sigma=\\sigma_{est}/2$)\n'
    'PSNR=%0.4g' % psnr_visushrink2)
ax[1, 2].imshow(im_visushrink4)
ax[1, 2].axis('off')
ax[1, 2].set_title(
    'Wavelet denoising\n(VisuShrink, $\\sigma=\\sigma_{est}/4$)\n'
    'PSNR=%0.4g' % psnr_visushrink4)

ax[0, 3].imshow(im_visushrink_soft)
ax[0, 3].axis('off')
ax[0, 3].set_title(
    'Wavelet denoising\n(VisuShrink Soft, $\\sigma=\\sigma_{est}$)\n'
    'PSNR=%0.4g' % psnr_visushrink_soft)

ax[1, 3].imshow(im_visushrink_hard)
ax[1, 3].axis('off')
ax[1, 3].set_title(
    'Wavelet denoising\n(VisuShrink Hard, $\\sigma=\\sigma_{est}$)\n'
    'PSNR=%0.4g' % psnr_visushrink_hard)
fig.tight_layout()

plt.savefig('denoising.png')
