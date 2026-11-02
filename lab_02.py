import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
# Read rice image
image = cv2.imread("rice.tif", cv2.IMREAD_GRAYSCALE)
# Display original image
plt.figure(figsize=(6, 6))
plt.imshow(image, cmap="gray")
plt.title("Original Rice Image")
plt.axis("off")
plt.show()
# Convert image to binary
_, binary = cv2.threshold(
    image, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
# Remove small noise
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# Label connected rice grains
label_image = label(binary > 0)
# Measure properties
properties = regionprops(label_image)
# Remove very small objects
properties = [p for p in properties if p.area > 50]
# Number of rice grains
number_of_rice = len(properties)
print("Number of rice grains:", number_of_rice)
print()
# Display measurements (using axis_major_length to avoid warnings)
print("Rice No.\tArea\t\tMajor Axis Length\tPerimeter")
print("-" * 65)
for i, prop in enumerate(properties, start=1):
    print(
        f"{i}\t\t"
        f"{prop.area:.2f}\t\t"
        f"{prop.axis_major_length:.2f}\t\t\t"
        f"{prop.perimeter:.2f}"
    )
min_area = 1000  
max_area = 10000
selected_rice = [
    p for p in properties
    if min_area <= p.area <= max_area
]
print()
print(f"Rice with area between {min_area} and {max_area}: "
      f"{len(selected_rice)}")
# Display selected rice grains
selected_image = np.zeros_like(label_image)
for p in selected_rice:
    selected_image[label_image == p.label] = 1
plt.figure(figsize=(6, 6))
plt.imshow(selected_image, cmap="gray")
plt.title(
    f"Rice with Area {min_area}-{max_area} pixels"
)
plt.axis("off")
plt.show()

