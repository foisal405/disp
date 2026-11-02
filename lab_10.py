#Perform image enhancement, smoothing and sharpening, in spatial domain using 
#different spatial filters and compare the performances 

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read image in grayscale mode
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found! Please check the file path.")
else:
    # --- A. SMOOTHING (Blurring) FILTERS ---
    
    # 1. Averaging Filter (Box Filter)
    avg_blur = cv2.blur(image, (5, 5))
    
    # 2. Gaussian Filter (Good for reducing Gaussian noise smoothly)
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 3. Median Filter (Best for removing Salt and Pepper noise)
    median_blur = cv2.medianBlur(image, 5)
    
    # 4. Laplacian Filter (High-pass filter for edge detection/sharpening)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    # 5. Custom Sharpening Kernel
    sharp_kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    sharpened_img = cv2.filter2D(image, -1, sharp_kernel)

    # --- C. DISPLAY COMPARISON USING SUBPLOTS ---
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(avg_blur, cmap="gray")
    plt.title("Smoothing: Averaging (Blur)")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(gaussian_blur, cmap="gray")
    plt.title("Smoothing: Gaussian Blur")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(median_blur, cmap="gray")
    plt.title("Smoothing: Median Filter")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(laplacian, cmap="gray")
    plt.title("Sharpening: Laplacian Filter")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(sharpened_img, cmap="gray")
    plt.title("Sharpening: Custom Kernel")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
