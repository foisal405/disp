#Display following image operation in MATLAB/Python - i) Threshold image ii) Power enhance contract image iii) High pass image. 

import cv2
import numpy as np
import matplotlib.pyplot as plt
# 1. Read image in grayscale mode
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Image not found! Please check the file path.")
else:
    # i) Threshold Image (Otsu Thresholding)
    _, thresh_img = cv2.threshold(
        image, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # ii) Power Enhance Contrast Image (Gamma Correction)
    gamma = 0.5 
    normalized_img = image / 255.0
    power_img = np.array(255 * (normalized_img ** gamma), dtype=np.uint8)
    # iii) High Pass Image (Custom High-Pass Filter)
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])
    high_pass_img = cv2.filter2D(image, -1, kernel)
    high_pass_img = cv2.normalize(high_pass_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Display operations using subplots
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(thresh_img, cmap="gray")
    plt.title("i) Threshold Image")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(power_img, cmap="gray")
    plt.title("ii) Power Enhance (Gamma) Image")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(high_pass_img, cmap="gray")
    plt.title("iii) High Pass Filter Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
