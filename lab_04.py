#Write a MATLAB/Python program to read an image and perform Lapliciant filter mask
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("images.jpg", cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
mask = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])
laplacian = cv2.filter2D(image, cv2.CV_64F, mask)
laplacian = np.absolute(laplacian)
laplacian = np.uint8(
    np.clip(laplacian, 0, 255)
)
plt.subplot(1, 2, 2)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian Filtered Image")
plt.axis("off")
plt.tight_layout()
plt.show()
