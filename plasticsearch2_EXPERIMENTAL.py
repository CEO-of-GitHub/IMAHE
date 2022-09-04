# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:55:00 2018

@author: Amirber

Use openCV and skimage to analyze microscope slide showing particles.
image source and previous work from:
    https://publiclab.org/notes/mathew/09-03-2015/opt

Parts are adapted from: https://peerj.com/articles/453/
"""
import cv2
from skimage import data, io, filters, feature, exposure, restoration
import numpy as np
import matplotlib.pyplot as plt
# Label image regions.
from skimage.measure import regionprops
import matplotlib.patches as mpatches
from skimage.morphology import label
from picamera import PiCamera
import time
import getch


imgPath = "microplastics4.jpg"
#imgPath = "circletest.png"
#imgPath = "frog s blood cells under 400x microscope.jpg"
#imgPath = 'C:/Users/Amirber/Documents/pm25/farc_snd6_(c6).jpg'
#imgPath = "test2.jpg"

camera = PiCamera()


# Set camera settings
camera.resolution = (1920, 1080)
camera.framerate = 15

camera.start_preview()
#time.sleep()

while True:
    inp = getch.getch()
    if inp == ' ':
        break

camera.capture(imgPath) 
camera.stop_preview()
camera.close()  




# Insert a um to pixel conversion ratio
um2pxratio = float(1000 / 680)
image = cv2.imread(imgPath,0)# data.coins()  # or any NumPy array!
#remove the 2 um scake
#image = image[:800,:] # limits the x-axis resolution of image to <800 so that file does not take too long to read



# CIRCULAR CROP IMAGE
"""
c_height = image.shape[0]
c_width = image.shape[1]
print("height: " + str(c_height) + " width: " + str(c_width))

cimg = image.copy()
#cimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, dp=1, minDist=2000, param1=20, param2=20, minRadius=0, maxRadius=0 ) # parameters: func(input, method for circle detection, inverse ratio, min distance between centers of detected circles, param1 (gradient value for edge detection), param2 (gradient threshold factor, lower value means more circles detected (more false positives)), minimum radius of circle to detect, maximum radius of circle to detect)
circles = np.uint16(np.around(circles))
for (x,y,r) in circles[0,:]:
    cv2.circle(cimg, (x,y), r, (0,255,0), 3)
    #cv2.circle(image, (x,y), r - (r * 0.8), (255,255,0), 3)

cv2.imshow("detected circles", cimg)
cv2.waitKey()
cv2.destroyAllWindows()
"""


# SQUARE CROP IMAGE

#cropped_img = image[825:1125,500:700]
image = image[350:850,750:1250]
#image = image[250:750,625:1125]
#image = image[210:710,700:1200]
#image = image[210:710,600:1100]

# SCIKIT IMAGE FILTER FUNCTIONS:

# GAMMA 
image = exposure.adjust_gamma(image, 1.75) # ideal value: 1.5-1.75

# LOGARITHMIC
image = exposure.adjust_log(image,1.25) # ideal value: 1.1-1.25

# DENOISE
image = restoration.denoise_tv_chambolle(image,weight=0.1) # ideal value: 0.1-0.2

# Show whole image
#io.imshow(image)
#plt.show()






#Show histogram
"""
values, bins = np.histogram(image,
                            bins=np.arange(256))
plt.figure()
plt.plot(bins[:-1], values)
plt.title("Image Histogram")
#plt.show()
"""

# Calculate Sobel edges
edges_sob = filters.sobel(image)
io.imshow(edges_sob)
plt.show()

# Show histogram of non-zero Sobel edges
"""
values, bins = np.histogram(np.nonzero(edges_sob) ,
                            bins=np.arange(1000))
plt.figure()
plt.plot(bins[:-1], values)
plt.title("Use Histogram to select thresholding value")
plt.show()
"""

# Using a threshold to binarize the images, condider replacing with an adaptice
# criteria. Raising the TH to 0.03 will remove the two touching particles but will 
# cause larger particles to split.
edges_sob_filtered = np.where(edges_sob>0.02,255,0) # SYNTAX : np.where(variable of condition (if array, function will go through and append each value), if condition true, if condition false)
#io.imshow(edges_sob_filtered)

#Use label on binary Sobel edges to find shapes
label_image = label(edges_sob_filtered)
fig,ax = plt.subplots(1,figsize=(20,10))
ax.imshow(image, cmap=plt.cm.gray)
ax.set_title('Labeled items', fontsize=24)
ax.axis('off')

#Do not plot regions smaller than 5 pixels on each axis
sizeTh=4



areaThmin = 100 # ideal value: 1000
#areaThmax = 100000
#region.area_bbox > areaThmin and region.area_bbox < areaThmax



for region in regionprops(label_image):
    # Draw rectangle around segmented coins.
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr),
                              maxc - minc,
                              maxr - minr,
                              fill=False,
                              edgecolor='red',
                              linewidth=2)
    """
    if region.area_bbox > areaThmin and region.area_bbox < areaThmax:
        #print(region.area,end="\n")
        print((maxr-minr)*(maxc-minc),end="\n")
    """
    if region.area_bbox > areaThmin:
        ax.add_patch(rect)

#Sort all found shapes by region size
sortRegions = [[(region.bbox[2]-region.bbox[0]) * (region.bbox[3] - region.bbox[1]),region.bbox] 
                for region in regionprops(label_image) if region.area_bbox > areaThmin]
sortRegions = sorted(sortRegions, reverse=True)

#Check particle sizes distribution
particleSize = [size[0] for size in sortRegions]

#Show histogram of non-zero Sobel edges
"""
plt.figure()
plt.plot(np.multiply(np.power(um2pxratio,2), particleSize),linewidth=2)
plt.xlabel('Particle count',fontsize=14)
plt.ylabel('Particle area',fontsize=14)
plt.title("Particle area distribution",fontsize=16)
"""
plt.show()


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

# ADDED CODE REFERENCES:
# https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_gamma
