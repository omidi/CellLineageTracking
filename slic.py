from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import skimage.color as color
from skimage import io
import matplotlib.pyplot as plt
import argparse
import numpy as np

def slic_segmentation(img, numSegments=1000, compactness=10):
    image = color.gray2rgb(img)
    # loop over the number of segments
    # apply SLIC and extract (approximately) the supplied number of segments
    segments = slic(image, n_segments=numSegments, sigma=1, max_iter=5, compactness=compactness)
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments, color=(0,1,1)))
    plt.axis("off")
    # show the plots
    plt.show()
from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

def hugh_circle_detection(image):
	# Load picture and detect edges
	edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

	fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))

	# Detect two radii
	hough_radii = np.arange(15, 30, 2)
	hough_res = hough_circle(edges, hough_radii)

	centers = []
	accums = []
	radii = []

	for radius, h in zip(hough_radii, hough_res):
		# For each radius, extract two circles
		num_peaks = 2
		peaks = peak_local_max(h, num_peaks=num_peaks)
		centers.extend(peaks)
		accums.extend(h[peaks[:, 0], peaks[:, 1]])
		radii.extend([radius] * num_peaks)

	# Draw the most prominent 5 circles
	image = color.gray2rgb(image)
	for idx in np.argsort(accums)[::-1][:5]:
		center_x, center_y = centers[idx]
		radius = radii[idx]
		cx, cy = circle_perimeter(center_y, center_x, radius)
		image[cy, cx] = (220, 20, 20)

	ax.imshow(image, cmap=plt.cm.gray)
	plt.show()


# def show_image(img):
# 	fig = plt.figure("image")
# 	ax = fig.add_subplot(1,1,1)
# 	ax.imshow(img)
# 	plt.axis("off")
# 	plt.show()

from skimage.filters import threshold_otsu, threshold_adaptive, rank

def otsu_thresholding(image):
	thresh = threshold_otsu(image)
	binary = image > thresh
	return binary


def adaptive_threshold(image):
	global_thresh = threshold_otsu(image)
	# binary_global = image > global_thresh
	block_size = 50
	binary_adaptive = threshold_adaptive(image, block_size, offset=10)
	# fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
	# ax0, ax1, ax2 = axes
	# plt.gray()
	# ax0.imshow(image)
	# ax0.set_title('Image')
	# ax1.imshow(binary_global)
	# ax1.set_title('Global thresholding')
	# ax2.imshow(binary_adaptive)
	# ax2.set_title('Adaptive thresholding')
	# for ax in axes:
	# 	ax.axis('off')
	# plt.show()
	return binary_adaptive


from skimage.util import img_as_ubyte
from skimage.morphology import disk, watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


def local_otsu(img):
	radius = 100
	selem = disk(radius)
	local_otsu = rank.otsu(img, selem)
	threshold_global_otsu = threshold_otsu(img)
	global_otsu = img >= threshold_global_otsu
	fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
	ax1, ax2, ax3, ax4 = ax.ravel()
	fig.colorbar(ax1.imshow(img, cmap=plt.cm.gray),
				 ax=ax1, orientation='horizontal')
	ax1.set_title('Original')
	ax1.axis('off')
	fig.colorbar(ax2.imshow(local_otsu, cmap=plt.cm.gray),
				 ax=ax2, orientation='horizontal')
	ax2.set_title('Local Otsu (radius=%d)' % radius)
	ax2.axis('off')
	ax3.imshow(img >= local_otsu, cmap=plt.cm.gray)
	ax3.set_title('Original >= Local Otsu' % threshold_global_otsu)
	ax3.axis('off')
	ax4.imshow(global_otsu, cmap=plt.cm.gray)
	ax4.set_title('Global Otsu (threshold = %d)' % threshold_global_otsu)
	ax4.axis('off')
	plt.show()


# def watershed_thresholding(image):
# 	# Generate the markers as local maxima of the distance to the background
# 	distance = ndi.distance_transform_edt(image)
# 	local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
# 								labels=image)
# 	markers = ndi.label(local_maxi)[0]
# 	labels = watershed(-distance, markers, mask=image)
# 	fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7), sharex=True, sharey=True,
# 							 subplot_kw={'adjustable': 'box-forced'})
# 	ax0, ax1, ax2 = axes
# 	ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
# 	ax0.set_title('Overlapping objects')
# 	ax1.imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
# 	ax1.set_title('Distances')
# 	ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
# 	ax2.set_title('Separated objects')
#
# 	for ax in axes:
# 		ax.axis('off')
#
# 	fig.tight_layout()
# 	plt.show()
#

from skimage import img_as_float
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

def denoising(astro):
	noisy = astro + 0.6 * astro.std() * np.random.random(astro.shape)
	noisy = np.clip(noisy, 0, 1)
	fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 5), sharex=True,
						   sharey=True, subplot_kw={'adjustable': 'box-forced'})

	plt.gray()

	ax[0, 0].imshow(noisy)
	ax[0, 0].axis('off')
	ax[0, 0].set_title('noisy')
	ax[0, 1].imshow(denoise_tv_chambolle(noisy, weight=0.1, multichannel=True))
	ax[0, 1].axis('off')
	ax[0, 1].set_title('TV')
	ax[0, 2].imshow(denoise_bilateral(noisy, sigma_range=0.05, sigma_spatial=15))
	ax[0, 2].axis('off')
	ax[0, 2].set_title('Bilateral')

	ax[1, 0].imshow(denoise_tv_chambolle(noisy, weight=0.2, multichannel=True))
	ax[1, 0].axis('off')
	ax[1, 0].set_title('(more) TV')
	ax[1, 1].imshow(denoise_bilateral(noisy, sigma_range=0.1, sigma_spatial=15))
	ax[1, 1].axis('off')
	ax[1, 1].set_title('(more) Bilateral')
	ax[1, 2].imshow(astro)
	ax[1, 2].axis('off')
	ax[1, 2].set_title('original')

	fig.tight_layout()

	plt.show()


from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy

def entropy_filtering(img):
	# # First example: object detection.
	#
	# noise_mask = 28 * np.ones((128, 128), dtype=np.uint8)
	# noise_mask[32:-32, 32:-32] = 30
	#
	# noise = (noise_mask * np.random.random(noise_mask.shape) - 0.5 *
	# 		 noise_mask).astype(np.uint8)
	# img = noise + 128

	entr_img = entropy(img, disk(10))

	fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 3))

	ax0.imshow(noise_mask, cmap=plt.cm.gray)
	ax0.set_xlabel("Noise mask")
	ax1.imshow(img, cmap=plt.cm.gray)
	ax1.set_xlabel("Noisy image")
	ax2.imshow(entr_img)
	ax2.set_xlabel("Local entropy")

	fig.tight_layout()

	# Second example: texture detection.

	image = img_as_ubyte(data.camera())

	fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4), sharex=True,
								   sharey=True,
								   subplot_kw={"adjustable": "box-forced"})

	img0 = ax0.imshow(image, cmap=plt.cm.gray)
	ax0.set_title("Image")
	ax0.axis("off")
	fig.colorbar(img0, ax=ax0)

	img1 = ax1.imshow(entropy(image, disk(5)), cmap=plt.cm.gray)
	ax1.set_title("Entropy")
	ax1.axis("off")
	fig.colorbar(img1, ax=ax1)

	fig.tight_layout()

	plt.show()


from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.morphology import reconstruction

def filter_bg(image):
	# Convert to float: Important for subtraction later which won't work with uint8
	image = img_as_float(color.gray2rgb(image))
	image = gaussian_filter(image, 1)
	seed = np.copy(image)
	seed[1:-1, 1:-1] = image.min()
	mask = image
	dilated = reconstruction(seed, mask, method='dilation')
	# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.5), sharex=True, sharey=True)
	# ax1.imshow(image)
	# ax1.set_title('original image')
	# ax1.axis('off')
	# ax1.set_adjustable('box-forced')
	# ax2.imshow(dilated, vmin=image.min(), vmax=image.max())
	# ax2.set_title('dilated')
	# ax2.axis('off')
	# ax2.set_adjustable('box-forced')
	# ax3.imshow(image - dilated)
	# ax3.set_title('image - dilated')
	# ax3.axis('off')
	# ax3.set_adjustable('box-forced')
	# fig.tight_layout()
	# plt.show()
	return image - dilate


from scipy import ndimage as ndi
from skimage import feature

def cany_edge(im):
    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(im)
    edges2 = feature.canny(im, sigma=3)
    # display results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)
    ax1.imshow(im, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('noisy image', fontsize=20)

    ax2.imshow(edges1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

    fig.tight_layout()

    plt.show()


from skimage import data
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.morphology import watershed
import cv2

def watershed_segmentation(image):
	# image = data.coins()
	image = cv2.medianBlur(image, 5)
	markers = np.zeros_like(image)
	markers[image < 30] = 1
	markers[image > 150] = 2
	elevation_map = sobel(image)
	segmentation = watershed(elevation_map, markers)
	fig, ax = plt.subplots(figsize=(4, 3))
	ax.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
	ax.axis('off')
	ax.set_title('segmentation')
	# plt.show()
	return elevation_map



from skimage import exposure
from skimage.filters import rank
from skimage.filters.rank import enhance_contrast_percentile, mean_bilateral


def color_enhancement(noisy_image):
	bilat = mean_bilateral(noisy_image.astype(np.uint16), disk(20), s0=10, s1=10)
	return bilat