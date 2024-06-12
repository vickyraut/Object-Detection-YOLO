import cv2

image = cv2.imread("../Image/image.jpg")

resize_image = cv2.resize(image, (0,0), fx = 0.2, fy = 0.2, interpolation = cv2.INTER_AREA)
cv2.imshow("Image", resize_image)

cv2.waitKey(0)
