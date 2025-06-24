import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read input image
image = cv2.imread('2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Input File
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.title("The input file")
plt.show()

#Convert to Grayscake format
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
plt.figure(figsize=(10,10))
plt.imshow(gray, cmap="gray")
plt.title("GrayScaled Image")
plt.show()

#For Edges
edge = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
plt.figure(figsize=(10,10))
plt.imshow(edge, cmap=("gray"))
plt.title("Edges")
plt.show

color = cv2.bilateralFilter(image, 9, 250, 250)
cartoon = cv2.bitwise_and(color, color, mask=edge)
plt.figure(figsize=(10,10))
plt.imshow(cartoon, cmap="gray")
plt.title("Cartoon Image")
plt.show()
