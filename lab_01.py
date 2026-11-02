#Display following image operation in MATLAB/Python - i) Histogram  image  ii) Low 
# pass filter mage  iii) High pass image.

import cv2
import numpy as np
import matplotlib.pyplot as plt
 
image_path = 'image.jpg' 
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
 
if img is None:
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(img, (150, 150), 80, 255, -1)
 
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
 
low_pass_img = cv2.GaussianBlur(img, (9, 9), 0)
 
high_pass_img = cv2.subtract(img, low_pass_img) 
 
plt.figure(figsize=(12, 8))
 
# Original Image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
 
# i) Histogram
plt.subplot(2, 2, 2)
plt.plot(hist, color='black')
plt.title('Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])

 
# ii) Low Pass Filtered Image
plt.subplot(2, 2, 3)
plt.imshow(low_pass_img, cmap='gray')
plt.title('Low Pass Filtered (Smoothed)')
plt.axis('off')
 
# iii) High Pass Filtered Image
plt.subplot(2, 2, 4)
plt.imshow(high_pass_img, cmap='gray')
plt.title('High Pass Filtered (Edges)')
plt.axis('off')
 
plt.tight_layout()
plt.show()

