import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Let user choose a file
Tk().withdraw()  # Hide the root tkinter window
file_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

# Read and convert to RGB
image = cv2.imread(file_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show input image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title("The Input File")
plt.axis('off')
plt.show()

# Convert to grayscale and blur
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray = cv2.medianBlur(gray, 5)

plt.figure(figsize=(10, 10))
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# Detect edges
edge = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    9, 9
)

plt.figure(figsize=(10, 10))
plt.imshow(edge, cmap="gray")
plt.title("Edges")
plt.axis('off')
plt.show()

# Smoothen the color image
color = cv2.bilateralFilter(image, 9, 250, 250)

# Combine edges and color
cartoon = cv2.bitwise_and(color, color, mask=edge)

plt.figure(figsize=(10, 10))
plt.imshow(cartoon)
plt.title("Cartoon Image")
plt.axis('off')
plt.show()

# Save the result
save_path = "cartoon_output.jpg"
cv2.imwrite(save_path, cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR))
print(f"Cartoon image saved as '{save_path}'")