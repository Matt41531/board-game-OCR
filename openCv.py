import cv2
import numpy as np

image = cv2.imread("Wingspan3.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imwrite("processed.png", thresh)
