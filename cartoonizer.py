import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

# Edge detection
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray = cv2.medianBlur(gray, 9)

# Combine adaptive threshold with Canny edges
edges_adaptive = cv2.adaptiveThreshold(
    gray_blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 2
)

edges_canny = cv2.Canny(gray_blur, 30, 100)
edges = cv2.bitwise_or(edges_adaptive, edges_canny)

#Fix broken edges
kernel = np.ones((2, 2), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

plt.axis("off")
plt.imshow(edges, cmap='gray')
plt.show()

# Flatter color region
k = 6
data = image.reshape(-1,3)
kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
image_reduced = kmeans.cluster_centers_[kmeans.labels_]
image_reduced = image_reduced.reshape(image.shape).astype(np.uint8)

# Smoothing with edge preservation
blurred = cv2.bilateralFilter(image_reduced, d=15, sigmaColor=80, sigmaSpace=80)

# Cartoon Styling
cartoon = blurred.copy()
cartoon[edges != 0] = [0, 0, 0]

# Texture
for i in range(0, cartoon.shape[0], 8):
    cv2.line(cartoon, (0, i), (cartoon.shape[1], i), (200,200,200), 1, cv2.LINE_AA)

#Display the comparison
plt.subplot(1, 2, 1)
plt.title("Original")
plt.axis("off")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Cartoonized")
plt.axis("off")
plt.imshow(cartoon)
plt.show()


# Save the result
save_path = "cartoon_output.jpg"
cv2.imwrite(save_path, cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR))
print(f"Cartoon image saved as '{save_path}'")