# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:55:00 2018
@author: Amirber
Use openCV and skimage to analyze microscope slide showing particles.
image source and previous work from:
    https://publiclab.org/notes/mathew/09-03-2015/opt
Parts are adapted from: https://peerj.com/articles/453/
"""
import math
import cv2
from PIL  import Image
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


imgPath = "microplastics3.jpg"
#imgPath = "imahe test.png"
#imgPath = "whitescreen.jpg"
#imgPath = "circletest.png"
#imgPath = "frog s blood cells under 400x microscope.jpg"
#imgPath = 'C:/Users/Amirber/Documents/pm25/farc_snd6_(c6).jpg'
#imgPath = "test2.jpg"

"""
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

"""


# Insert a um to pixel conversion ratio
um2pxratio = float(1000 / 680)

"""
pil_image_1 = Image.open(imgPath)
numpy_image = np.array(pil_image_1)
image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
"""

image = cv2.imread(imgPath,0)# data.coins()  # or any NumPy array!
#imgg = cv2.imread(imgPath, 1)
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
#image = image[350:850,800:1300] # original
#image = image[250:750,625:1125]
image = image[210:710,600:1100]

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



areaThmin = 1000 # ideal value: 1000
#areaThmax = 100000
#region.area_bbox > areaThmin and region.area_bbox < areaThmax



for region in regionprops(label_image):
    # Draw rectangle around segmented coins.
    minr, minc, maxr, maxc = region.bbox # SYNTAX : [minimum y value, minimum x value, maximum y value, maximum x value] = region.bbox
                                         # minc to maxc is the x-pixel length, while minr to maxr is the y-pixel height
    rect = mpatches.Rectangle((minc, minr), # SYNTAX : mpatches.Rectangle((x coordinate, y coordinate), pixel width to add to x coordinate, pixel height to add to y coordinate, angle (though not used in this case))
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
#gumaganern?

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
    minr, minc, maxr, maxc = region[1] # minr = minimum rows ; minc = minimum columns ; maxr = maximum rows ; maxc = maximum columns
    
    fig, ax = plt.subplots(1,3,figsize=(15,6)) # SYNTAX : plt.subplots(no. of rows, no. of columns, specified cell/position of object)
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
    ax[1].plot([0.1*(maxc - minc), 0.3*(maxc - minc)], # SYNTAX: plt.plot(x domain of graph, y range of graph, modifier) 
             [0.9*(maxr - minr),0.9*(maxr - minr)],'r')
    ax[1].text(0.15*(maxc - minc), 0.87*(maxr - minr), # the plotted line shown is not the measurement of longest length of microplastic particle, but rather an arbitrarily placed line on the image which serves as a measuring stick for the conversion of pixels to um. "[0.1*(maxc - minc), 0.3*(maxc - minc)" refers to the measure of the line which is plotted in the image while "str(round(0.2*(maxc - minc)*um2pxratio,1))+'um'" is the text that shows its measurements
          str(round(0.2*(maxc - minc)*um2pxratio,1))+'um',
          color='red', fontsize=12, horizontalalignment='center')
          
    ax[2].imshow(edges_sob_filtered[minr:maxr,minc:maxc],cmap='gray')
    ax[2].set_title('Edge view', fontsize=16)
    ax[2].axis("off")
    
    # convex hull function finds points pointing the convex hull edge pixels of sobel image
        # for loop of: pythagorean c measurement of any two given points from each pixel found by the convex hull function, iterating x!/(x-2)! times, wherein x is the number of edge pixels
            # if: found pythagorean c is larger than previous largest pythagorean c
                # replace largest pythagorean c variable
    # convhull could return area of convex hull, meaning area of microplastic particle could be measured

    sobel_image = edges_sob_filtered[minr:maxr,minc:maxc]
    sobel_edges = []
    rows = sobel_image.shape[0] # height, y
    cols = sobel_image.shape[1] # width, x
    #print(rows)
    #print(cols)
    
    # REMINDER ABT SYNTAX FOR CROPPING IMAGES: image[y,x]
    # convert from openCV2 to PIL. Notice the COLOR_BGR2RGB which means that 
    # color is converted from BGR to RGB
    
    #color_converted = cv2.cvtColor(sobel_image, cv2.COLOR_BGR2RGB)
    #sobel_image = Image.fromarray(color_converted)
    
    


    # 1: Find the coordinates of all white pixels
    # 2: Perform pythagorean calculations on all possible combinations of the white pixels to find the longest length








    # STEP 1
    
    whites_detected = 0
    i = 0
    while i < rows:
        j = 0
        while j < cols:
            if sobel_image[i,j] != 0: # if pixel is not black
                coords = [i,j]
                sobel_edges.append(coords)
                #print("white", j, ",", i)
                whites_detected += 1
            #else:
                #print("black")
            j += 1
        i += 1
    #print("There are", whites_detected, "white pixels detected.")









    # STEP 2
    
    longest_length = 0 # longest length measurement
    ll_coords = [] # coordinates of longest length
    i = 0
    while i < whites_detected:
        while j < whites_detected:
            
            y_val = abs(sobel_edges[i][0] - sobel_edges[j][0])
            x_val = abs(sobel_edges[i][1] - sobel_edges[j][1])
            
            pythagorean_measure = round(math.sqrt((x_val)**2 + (y_val)**2))
            if pythagorean_measure > longest_length:
                longest_length = pythagorean_measure
                coords_used = [sobel_edges[j],sobel_edges[i]]
                if len(ll_coords) != 0:
                    ll_coords.pop(0)
                ll_coords.append(coords_used)
                #print(longest_length)
            
            j += 1
        i += 1
   
    print("Longest length found:", longest_length)
    print("Coordinates (y,x):", ll_coords)
    print("Converted to micrometers: ", longest_length*um2pxratio, "um")



    # DRAW RED LINE ON LONGEST LENGTH COORDINATES
    ax[2].text(0.075*(maxc - minc), 0.87*(maxr - minr), # the plotted line shown is not the measurement of longest length of microplastic particle, but rather an arbitrarily placed line on the image which serves as a measuring stick for the conversion of pixels to um. "[0.1*(maxc - minc), 0.3*(maxc - minc)" refers to the measure of the line which is plotted in the image while "str(round(0.2*(maxc - minc)*um2pxratio,1))+'um'" is the text that shows its measurements
          'longest length: '+str(round(longest_length*um2pxratio))+'um',
          color='red', fontsize=12, horizontalalignment='left')
    ax[2].plot([ll_coords[0][0][1],ll_coords[0][1][1]],[ll_coords[0][0][0],ll_coords[0][1][0]],'r')











    """
    for i in range(rows):
        for j in range(cols):
            if sobel_image[j,i] != 0: # if pixel is not black
                coords = [j,i]
                sobel_edges.append(coords)
    print(sobel_edges)
    """
    """
    while i < rows:
        while j < cols:
            if sobel_image[j,i] != 0: # if pixel is not black
                coords = [j,i]
                sobel_edges.append(coords)
                print("corn")
            else:
                print("pop")
    #print(sobel_edges)
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    conv_edges = convhull(sobel_edges)
    longest_length = 0
    for i in range(rows2):
        for j in range(cols2):
            if #current pixel iteration is part of conv_edges
                # for loop iterating through conv_edges pixels again
                    #pythagorean c calculation
    
    
    print(image[minr:maxr,minc:maxc])
    print(sobel_image)
    print()
    """
    
    """
    
    for pixel in sobel_image:
        if sobel_image[:,:,1] == 255:
            if sobel_image[:,:,2] == 255:
                if sobel_image[:,:,3] == 255:
                    sobel_edges.append(pixel)
    print(sobel_edges)
    """
        
    plt.show()

"""    
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
"""   

    
#Add fractal dimension estimation https://github.com/scikit-image/scikit-image/issues/1730

# ADDED CODE REFERENCES:
# https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_gamma
