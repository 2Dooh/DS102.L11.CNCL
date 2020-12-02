import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./data/video20-305/1.jpg')
resized_img = cv.resize(img, (640, 360), interpolation=cv.INTER_LINEAR)
cv.imwrite('./resized_1_640-360.jpg', resized_img)
plt.imshow(resized_img)
plt.show()