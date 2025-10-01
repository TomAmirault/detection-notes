import numpy as np
import ot
import cv2

img1 = cv2.imread("/Users/noahparisse/Downloads/Detection-de-feuilles/test/images/notebook-2637757__340-1-_jpg.rf.c99d2b74df2359ea917c461425c728a4.jpg")
img2 = cv2.imread("/Users/noahparisse/Downloads/Detection-de-feuilles/test/images/images-3-_jpg.rf.8b995a67da5b5c07cd1f51a432c88f68.jpg")
# print(img1.shape) -> (640, 640, 3)
# print(img2.shape)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# print(gray1.shape) -> (640, 640)
# print(gray2.shape)

# Solve the OT problem
sol = ot.solve_sample(gray1, gray2, a, b)

# get the OT plan
P = sol.plan

# get the OT loss
loss = sol.value

# get the dual potentials
alpha, beta = sol.potentials

print(f"OT loss = {loss:1.3f}")