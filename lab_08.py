import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

# 1. Read 'coins.png' image in grayscale mode
#https://www.davidduggleby.com/auctions/2282/Coins-Banknotes/102900/
image = cv2.imread("images(2).jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: 'coins.png' file not found! Please check the image name and path.")
else:
    # 2. Convert image to binary using Otsu's Thresholding
    _, binary = cv2.threshold(
        image, 0, 255, 
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove small background noise using morphological opening
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 3. Label connected components (Coins)
    label_image = label(binary > 0)

    # 4. Measure region properties to get the area of each coin
    properties = regionprops(label_image)

    # Filter out very small objects (noise)
    properties = [p for p in properties if p.area > 50]

    # Total number of coins detected
    total_coins = len(properties)
    print(f"Total Coins Detected: {total_coins}\n")

    # Display measurements table
    print("Coin No.\tArea (Pixels)")
    print("-" * 30)

    for i, prop in enumerate(properties, start=1):
        print(f"{i}\t\t{prop.area:.2f}")

    # 5. Display Original and Labeled Coin Images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Coins Image")
    axes[0].axis("off")

    axes[1].imshow(label_image, cmap="nipy_spectral")
    axes[1].set_title(f"Labeled Coins (Total: {total_coins})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
