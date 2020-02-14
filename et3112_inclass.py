#!/usr/bin/env python
# coding: utf-8

# # Reading and Writing Images

# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 as cv
import matplotlib.pyplot as plt

filename = './images/emma.jpg'
img = cv.imread(filename, cv.IMREAD_COLOR)
img_show = cv.cvtColor(img, cv.COLOR_BGR2RGB)

img = cv.add(img, 40)
cv.imwrite('./results/emma_processed.jpg', img)

fig, ax = plt.subplots(1, figsize =(6,6))
ax.imshow(img_show)
plt.show()


# In[9]:


print(img.shape)


# In[11]:


import numpy as np
a = np.array([1,2,3, 4])
print(a)


# # Gaussian Noise Removal

# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

fn = './images/hugh.jpg'
imgo = cv.imread(fn, cv.IMREAD_COLOR)
mu = 0
sigma = 20
noise = np.array(imgo.shape).astype('uint8')
noise = np.random.normal(mu, sigma, imgo.shape)
noise = noise.reshape(imgo.shape)
noisy_image = cv.add(imgo, noise, dtype=cv.CV_8UC3)
filtered = cv.GaussianBlur(noisy_image, (5,5), 2)

# Color conversions to be able to show in matplotlib
noisy_image_show = cv.cvtColor(noisy_image, cv.COLOR_BGR2RGB)
imgo_show = cv.cvtColor(imgo, cv.COLOR_BGR2RGB)
filtered_show = cv.cvtColor(filtered, cv.COLOR_BGR2RGB)

fig, ax = plt.subplots(3, figsize = (12,6))
ax[0].imshow(imgo_show)
ax[0].set_title('Original')
ax[1].imshow(noisy_image_show)
ax[1].set_title('Noise Added')
ax[2].imshow(filtered_show)
ax[2].set_title('Filtered')
plt.show()


# In[16]:


print(noise.shape)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 as cv
import matplotlib.pyplot as plt

fn = './images/hugh.jpg'
imgo = cv.imread(fn, cv.IMREAD_COLOR)

fig, ax = plt.subplots()
ax.imshow(imgo)
plt.show()

print(imgo.shape)
print(imgo.dtype)
print(imgo.size)


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 as cv
import matplotlib.pyplot as plt

fn = './images/hugh.jpg'
im1 = cv.imread(fn, cv.IMREAD_GRAYSCALE)
im2 = im1 + 100
im3 = cv.add(im1,100)
fig, ax = plt.subplots(1,3)
ax[0].imshow(im1, cmap='gray')
ax[1].imshow(im2, cmap='gray')
ax[2].imshow(im3, cmap='gray')
plt.show()


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def image_brighten(image, shift):
    h = image.shape[0]
    w = image.shape[1]
    result = np.zeros((h,w), np.uint8)
    for i in range(0,h):
        for j in range(0,w):
            result[i,j] = image[i,j] + 100
    return result

fn = './images/hugh.jpg'
im1 = cv.imread(fn, cv.IMREAD_GRAYSCALE)

im2 = image_brighten(im1, 100)
fig, ax = plt.subplots(1,2, figsize=(6,6))
ax[0].imshow(im1, cmap='gray')
ax[1].imshow(im2, cmap='gray')
# ax[2].imshow(im3, cmap='gray')
plt.show()


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
transform = np.arange(0,256).astype('uint8')
transform = np.arange(255, -1, -1).astype('uint8')
fig, ax = plt.subplots()
ax.plot(transform)
ax.set_xlabel(r'Input, $f(\mathbf{x})$')
ax.set_ylabel('Output, $\mathrm{T}[f(\mathbf{x})]$')
ax.set_xlim(0,255)
ax.set_ylim(0,255)
ax.set_aspect('equal')
plt.show()
img_orig = cv.imread('./images/katrina.jpg', cv.IMREAD_GRAYSCALE)
print(img_orig.shape)

cv.namedWindow('Image', cv.WINDOW_AUTOSIZE)
cv.imshow('Image', img_orig)
cv.waitKey(0)
image_transformed = cv.LUT(img_orig, transform)

cv.imshow('Image', image_transformed)
cv.waitKey(0)
cv.destroyAllWindows()


# In[3]:


print(transform.shape)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('./images/katrina.jpg', cv.IMREAD_GRAYSCALE)
hist = cv.calcHist([img], [0], None, [256], [0,256])

plt.plot(hist)
plt.show()


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

im1 = cv.imread('./images/katniss.jpg')
im1 = cv.cvtColor(im1, cv.COLOR_BGR2RGB)
gamma = 0.2
transform = (np.arange(0,256)/255.)**gamma

im2 = cv.LUT(im1, transform)*255
im2 = im2.astype('uint8')

fig, ax = plt.subplots(1,2, figsize=(10,10))
ax[0].imshow(im1)
ax[1].imshow(im2)
plt.show()


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

im1 = cv.imread('./images/maya.jpg', cv.IMREAD_REDUCED_COLOR_2)

kernel = np.array([[1./9., 1./9., 1./9.], [1./9., 1./9., 1./9.], [1./9., 1./9., 1./9.]])
kernel = 1/25.*np.ones((5,5))

im2 = cv.filter2D(im1, cv.CV_8U, kernel)

cv.namedWindow('Image', cv.WINDOW_AUTOSIZE)
cv.imshow('Image', im1)
cv.waitKey(0)

cv.imshow('Image', im2)
cv.waitKey(0)
cv.destroyAllWindows()


# In[29]:


print(kernel.shape)
print(sum(sum(kernel)))


# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

im1 = cv.imread('./images/building.tif', cv.IMREAD_GRAYSCALE)

kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

im2 = cv.filter2D(im1, cv.CV_8U, kernel)
im2 = cv.normalize(im2 + 127, im2, 255, 0, cv.NORM_MINMAX)

cv.namedWindow('Image', cv.WINDOW_AUTOSIZE)
cv.imshow('Image', im1)
cv.waitKey(0)

cv.imshow('Image', im2)
cv.waitKey(0)
cv.destroyAllWindows()


# In[44]:


# Sharpening
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import stats

img = cv.imread('./images/tom.jpg', cv.IMREAD_REDUCED_GRAYSCALE_2)

kernel = 1/25.*np.ones((5,5))
blurred = cv.filter2D(img, cv.CV_8U, kernel)
diff = img.astype('float32') - blurred.astype('float32')
sharpened = cv.addWeighted(img.astype('float32') , 1.0, diff, 1.5, 0)



fig, axes  = plt.subplots(2,2, sharex='all', sharey='all', figsize=(18,10))
axes[0,0].imshow(img, cmap='gray')
axes[0,0].set_title('Original')
axes[0,0].set_xticks([]), axes[0,0].set_yticks([])
axes[0,1].imshow(blurred, cmap='gray')
axes[0,1].set_title('Blurred')
axes[0,1].set_xticks([]), axes[0,1].set_yticks([])
axes[1,0].imshow(diff + 125, cmap='gray')
axes[1,0].set_title('Difference')
axes[1,0].set_xticks([]), axes[1,0].set_yticks([])
axes[1,1].imshow(sharpened, cmap='gray')
axes[1,1].set_title('Sharpened')
axes[1,1].set_xticks([]), axes[1,1].set_yticks([])
plt.show()


# In[ ]:




