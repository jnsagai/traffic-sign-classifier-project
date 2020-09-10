# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:46:24 2020

@author: jnnascimento
"""

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

#Load the images on a new test dataset
X_NewTestImages = np.zeros((5,32,32,3), dtype = np.float32)
y_newTestImages = np.array([16,35,42,40,23])

X_NewTestImages[0] = mpimg.imread("test_images/00161.png")
X_NewTestImages[1] = mpimg.imread("test_images/00260.png")
X_NewTestImages[2] = mpimg.imread("test_images/00315.png")
X_NewTestImages[3] = mpimg.imread("test_images/00499.png")
X_NewTestImages[4] = mpimg.imread("test_images/00565.png")

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20,10))
ax1.set_title('Random training')
ax1.imshow(X_NewTestImages[0])

ax2.set_title('Random training')
ax2.imshow(X_NewTestImages[1])

ax3.set_title('Random training')
ax3.imshow(X_NewTestImages[2])

ax4.set_title('Random training')
ax4.imshow(X_NewTestImages[3])

ax5.set_title('Random training')
ax5.imshow(X_NewTestImages[4])