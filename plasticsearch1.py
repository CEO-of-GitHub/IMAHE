# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:55:00 2018

@author: Amirber

Use openCV and skimage to analyze microscope slide showing particles.
image source and previous work from:
    https://publiclab.org/notes/mathew/09-03-2015/opt

Parts are adapted from: https://peerj.com/articles/453/
"""
from cv2 import imread, morphologyEx
from skimage import data, io, filters, feature
import numpy as np
import matplotlib.pyplot as plt
# Label image regions.
from skimage.measure import regionprops
import matplotlib.patches as mpatches
from skimage.morphology import label

imgPath = 'C:/Users/Amirber/Documents/pm25/farc_snd6_(c6).jpg'
# Insert a um to pixel conversion ratio
um2pxratio = 1
image = imread(imgPath,0)# data.coins()  # or any NumPy array!
#remove the 2 um scake
image = image[:800,:]
#Whoe image
io.imshow(image)

#Show histogram
values, bins = np.histogram(image,
                            bins=np.arange(256))
plt.figure()
plt.plot(bins[:-1], values)
plt.title("Image Histogram")
plt.show()

#Calculate Soble edges
edges_sob = filters.sobel(image)
io.imshow(edges_sob)

#Show histogram of non-sero Sobel edges
values, bins = np.histogram(np.nonzero(edges_sob) ,
                            bins=np.arange(1000))
plt.figure()
plt.plot(bins[:-1], values)
plt.title("Use Histogram to select thresholding value")

#Using a threshold to binarize the images, condider replacing with an adaptice
# criteria. raing the TH to 0.03 will remove the two tuching particles but will 
#cause larger oarticles to split.
edges_sob_filtered = np.where(edges_sob>0.01,255,0)
io.imshow(edges_sob_filtered)


#Use lable on binnary Sobel edges to find shapes
label_image = label(edges_sob_filtered)
fig,ax = plt.subplots(1,figsize=(20,10))
ax.imshow(image, cmap=plt.cm.gray)
ax.set_title('Labeled items', fontsize=24)
ax.axis('off')

#Do not plot regions smaller thn 5 pixels on each axis
sizeTh=4

for region in regionprops(label_image):
    # Draw rectangle around segmented coins.
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr),
                              maxc - minc,
                              maxr - minr,
                              fill=False,
                              edgecolor='red',
                              linewidth=2)
    ax.add_patch(rect)

#Sort all found shapes by region size
sortRegions = [[(region.bbox[2]-region.bbox[0]) * (region.bbox[3] - region.bbox[1]),region.bbox] 
                for region in regionprops(label_image) if
                ((region.bbox[2]-region.bbox[0])>sizeTh and (region.bbox[3] - region.bbox[1])>sizeTh)]
sortRegions = sorted(sortRegions, reverse=True)

#Check particle sizes distribution
particleSize = [size[0] for size in sortRegions]

#Show histogram of non-sero Sobel edges
plt.figure()
plt.plot(np.multiply(np.power(um2pxratio,2), particleSize),linewidth=2)
plt.xlabel('Particle count',fontsize=14)
plt.ylabel('Particle area',fontsize=14)
plt.title("Particle area distribution",fontsize=16)
#Show 5 largest regions location, image and edge
for region in sortRegions[:5]:
    # Draw rectangle around segmented coins.
    minr, minc, maxr, maxc = region[1]
    fig, ax = plt.subplots(1,3,figsize=(15,6))
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('full frame', fontsize=16)
    ax[0].axis('off')
    rect = mpatches.Rectangle((minc, minr),
                          maxc - minc,
                          maxr - minr,
                          fill=False,
                          edgecolor='red',
                          linewidth=2)
    ax[0].add_patch(rect)

    ax[1].imshow(image[minr:maxr,minc:maxc],cmap='gray')
    ax[1].set_title('Zoom view', fontsize=16)
    ax[1].axis("off")
    ax[1].plot([0.1*(maxc - minc), 0.3*(maxc - minc)],
             [0.9*(maxr - minr),0.9*(maxr - minr)],'r')
    ax[1].text(0.15*(maxc - minc), 0.87*(maxr - minr),
          str(round(0.2*(maxc - minc)*um2pxratio,1))+'um',
          color='red', fontsize=12, horizontalalignment='center')

    ax[2].imshow(edges_sob_filtered[minr:maxr,minc:maxc],cmap='gray')
    ax[2].set_title('Edge view', fontsize=16)
    ax[2].axis("off")
    plt.show()
    
#Show 5 smallest regions location, image and edge
for region in sortRegions[-5:]:
    # Draw rectangle around segmented coins.
    minr, minc, maxr, maxc = region[1]
    fig, ax = plt.subplots(1,3,figsize=(15,6))
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('full frame', fontsize=16)
    ax[0].axis('off')
    rect = mpatches.Rectangle((minc, minr),
                          maxc - minc,
                          maxr - minr,
                          fill=False,
                          edgecolor='red',
                          linewidth=2)
    ax[0].add_patch(rect)

    ax[1].imshow(image[minr:maxr,minc:maxc],cmap='gray')
    ax[1].set_title('Zoom view', fontsize=16)
    ax[1].axis("off")
    ax[1].plot([0.1*(maxc - minc), 0.3*(maxc - minc)],
             [0.9*(maxr - minr),0.9*(maxr - minr)],'r')
    ax[1].text(0.15*(maxc - minc), 0.87*(maxr - minr),
          str(round(0.2*(maxc - minc)*um2pxratio,1))+'um',
          color='red', fontsize=12, horizontalalignment='center')

    ax[2].imshow(edges_sob_filtered[minr:maxr,minc:maxc],cmap='gray')
    ax[2].set_title('Edge view', fontsize=16)
    ax[2].axis("off")
    plt.show()
    
#Add fractal dimension estimation https://github.com/scikit-image/scikit-image/issues/1730
