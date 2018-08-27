import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.animation as animation
import math

# create the new distribution of pixels
class Equalizer:

	def __init__(self, img_data = None):
		self.data = img_data

	def dist(self, hist, total) :
		dyn = np.empty(256, dtype=np.float32)
		sumn = 0.0
		sumn_1 = 0.0
		for i in range(256):
			sumn = hist[i] + sumn_1
			dyn[i] = math.floor((255)*sumn/total)
			sumn_1 = sumn
		return dyn

	# def prob_n(n , hist, total) :
	# 	return hist[n]/total

	def equalize_gray(self, img): 
		hist = cv2.calcHist([img],[0],None,[256],[0,256])
		height, width = img.shape
		total = height * width
		dyn = self.dist(hist, total)
		dst = np.empty((height, width), dtype=np.float32)
		for i in range(height): 
			for j in range(width):
				pix = img[i, j]
				dst[i, j] = dyn[pix]
		return dst

	def equalize_color(self, img):
		# height, width, x = img.shape
		img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		dst = img1[:, :, 2]
		eq = self.equalize_gray(dst) 
		img1[:, :, 2] = eq
		return cv2.cvtColor(img1, cv2.COLOR_HSV2RGB)

# ref_file = 'tree'
# ref_file = 'clown'
# ref_file = 'guitarist'
# ref_file = 'fox'																																																																																																																																																																
# ref_file = 'shape'
# ref_file = 'chemise'
# ref_file = 'city'
# ref_file = 'parasol'
ref_file = 'under_expose'

# file = 'test'
# file = 'tree'
# file = 'clown'
# file = 'guitarist'
# file = 'fox'																																																																																																																																																																
# file = 'shape'
# file = 'chemise'
# file = 'city'
# file = 'parasol'
file = 'under_expose'
	
ref = cv2.imread('images/' + ref_file + '.jpg', 0)
src = cv2.imread('images/' + file + '.jpg', 0)
src_c = cv2.imread('images/' + file + '.jpg', 1)

hist_ref = cv2.calcHist([ref],[0],None,[256],[0,256])
hist_src = cv2.calcHist([src],[0],None,[256],[0,256])

hsv = cv2.cvtColor(src_c, cv2.COLOR_BGR2HSV)

eq = Equalizer()

dst = eq.equalize_gray(src)
dst1 = eq.equalize_color(src_c)

# ref_b = cv2.bilateralFilter(ref, 15, 80, 80)
# src_b = cv2.bilateralFilter(src, 15, 80, 80)

f = plt.figure(1, figsize=(20,8))
plt.suptitle('Demonstration of Histogram Equalization', fontsize=16)
ax1 = f.add_subplot(221)
ax2 = f.add_subplot(222)
ax3 = f.add_subplot(223)
ax4 = f.add_subplot(224)
ax1.imshow(cv2.cvtColor(src_c, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')
ax2.hist(src.ravel(), 256, [0, 256])
ax2.set_title('Histogram of Pixel Values (in HSV)')
ax3.imshow(dst1)
ax3.set_title('Image After Histogram Equalization')
ax4.hist(dst1.ravel(), 256, [0, 256])
ax4.set_title('Equalized Histogram of Pixel Values (in HSV)')
ax4.set_xlabel('Pixel Value')

plt.show()