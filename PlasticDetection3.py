# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:55:00 2018

@author: Amirber

Use openCV and skimage to analyze microscope slide showing particles.
image source and previous work from:
   https://publiclab.org/notes/amirberAgain/01-12-2018/python-and-opencv-to-analyze-microscope-slide-images-of-airborne-particles

Parts are adapted from: https://peerj.com/articles/453/
"""
import cv2, sys
from cv2 import imread, morphologyEx
from skimage import data, io, filters, feature
import numpy as np
import matplotlib.pyplot as plt
# Label image regions.
from skimage.measure import regionprops
import matplotlib.patches as mpatches
from skimage.morphology import label


imgPath = sys.argv[1]
# Insert a um to pixel conversion ratio
um2pxratio = 1
image = imread(imgPath,-1)# data.coins()  # or any NumPy array!
#image = cv2.pyrDown(cv2.pyrDown(image))
image = cv2.pyrDown(image)
#Show whoe image
plt.figure()
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#%%
#Show whoe image
plt.figure(figsize=(6,6))
plt.imshow(hsv[:50,:50,:])
plt.axis('off')
#%%
edges_can = hsv.copy()
#Calculate Soble edges on each HSV color channel
edges_can[:,:,0] = feature.canny(hsv[:,:,0]/255.)# filters.sobel(hsv[:,:,0]) # 
edges_can[:,:,1] = feature.canny(hsv[:,:,1]/255.)#filters.sobel(hsv[:,:,1]) # 
edges_can[:,:,2] = feature.canny(hsv[:,:,2]/255.)#filters.sobel(hsv[:,:,2]) # 

fig,ax = plt.subplots(1,4, figsize=(16,8))
ax[0].imshow(edges_can[:,:,0])
ax[1].imshow(edges_can[:,:,1])
ax[2].imshow(edges_can[:,:,2])
ax[3].imshow(image[:,:,1]+255*edges_can[:,:,1],cmap='gray')
ax[0].axis('off')
plt.show()
#%%
#edges_can = np.true_divide(np.mean(edges_can[:,:,1],edges_can[:,:,2]),2)
#Show histogram of non-sero Sobel edges
values, bins = np.histogram(np.nonzero(edges_can[:,:,1]) ,
                            bins=np.arange(255))
plt.figure()
plt.plot(bins[:-1], values)
plt.title("Use Histogram to select thresholding value")
plt.show()

#Using a threshold to binarize the images, condider replacing with an adaptice
# criteria. raing the TH to 0.03 will remove the two tuching particles but will 
#cause larger oarticles to split.
edges_can_filtered_h = np.where(edges_can[:,:,0]>20/255.,255,0)
edges_can_filtered_s = np.where(edges_can[:,:,1]>20/255.,255,0)
edges_can_filtered_v = np.where(edges_can[:,:,2]>20/255.,255,0)

fig,ax = plt.subplots(1,4, figsize=(16,8))
ax[0].imshow(np.where(edges_can[:,:,0]>20/255.,255,0))
ax[1].imshow(np.where(edges_can[:,:,1]>20/255.,255,0))
ax[2].imshow(np.where(edges_can[:,:,2]>20/255.,255,0))
ax[3].imshow(image[:,:,1]+255*edges_can[:,:,1],cmap='gray')
ax[0].axis('off')
plt.show()
#%%


#Use lable on binnary Sobel edges to find shapes
label_image0 = label(edges_can_filtered_h)
label_image1 = label(edges_can_filtered_s)
label_image2 = label(edges_can_filtered_v)
#label_image = np.v
fig,ax = plt.subplots(1,figsize=(20,20))
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap=plt.cm.gray)
ax.set_title('Labeled items', fontsize=24)
ax.axis('off')

#Do not plot regions smaller thn 5 pixels on each axis
sizeTh=4

for region in regionprops(label_image0):
    # Draw rectangle around segmented coins.
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr),
                              maxc - minc,
                              maxr - minr,
                              fill=False,
                              edgecolor='red',
                              linewidth=2)
    
for region in regionprops(label_image1):
    # Draw rectangle around segmented coins.
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr),
                              maxc - minc,
                              maxr - minr,
                              fill=False,
                              edgecolor='green',
                              linewidth=2)
    ax.add_patch(rect)
for region in regionprops(label_image2):
    # Draw rectangle around segmented coins.
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr),
                              maxc - minc,
                              maxr - minr,
                              fill=False,
                              edgecolor='blue',
                              linewidth=2)
    ax.add_patch(rect)
