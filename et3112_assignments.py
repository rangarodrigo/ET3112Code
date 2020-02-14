#!/usr/bin/env python
# coding: utf-8

# # Rice Image

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
fn = './images/rice.png'
im = cv.imread(fn, cv.IMREAD_GRAYSCALE)

kernel = np.ones((13,13),np.uint8)
bg = cv.morphologyEx(im, cv.MORPH_OPEN, kernel)
bgcorrected  = cv.addWeighted(im, 1, bg, -1, 0)
bgcorrected = cv.normalize(bgcorrected, bgcorrected, 255, 0, cv.NORM_MINMAX)
thresh = cv.inRange(bgcorrected, 50, 255)
kernel = np.ones((3,3),np.uint8)
thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
retval, labels, stats, centroids = cv.connectedComponentsWithStats(thresh)
print(retval)
colomapped = cv.applyColorMap((labels/np.amax(labels)*255).astype('uint8'), cv.COLORMAP_PARULA)


cv.namedWindow('Original', cv.WINDOW_AUTOSIZE)
cv.imshow('Original', im)
cv.waitKey(0)

cv.namedWindow('Background', cv.WINDOW_AUTOSIZE)
cv.imshow('Background', bg)
cv.waitKey(0)

cv.namedWindow('BgCorrected', cv.WINDOW_AUTOSIZE)
cv.imshow('BgCorrected', bgcorrected)
cv.waitKey(0)

cv.namedWindow('Thresh', cv.WINDOW_AUTOSIZE)
cv.imshow('Thresh', thresh)
cv.waitKey(0)

cv.namedWindow('Labels', cv.WINDOW_AUTOSIZE)
cv.imshow('Labels', colomapped)
cv.waitKey(0)

cv.destroyAllWindows()


 


# In[11]:


print(np.amax(labels)) 
print(np.amin(labels))


# In[2]:


print(thresh.dtype)


# In[12]:


from sympy import *
a, z, x = symbols('a z x')
z = (2  - 5/6*a)/(1 - 5/6*a + 1/6*a**2)
print(apart(z))


# In[ ]:




