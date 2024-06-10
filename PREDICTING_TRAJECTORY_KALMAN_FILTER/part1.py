import cv2
from kalmanfilter import KalmanFilter

kf = KalmanFilter()

circle_positions = [(50, 100), (100, 100), (150, 100), (200, 100), (250, 100), (300, 100), (350, 100), (400, 100), (450, 100)]
img = cv2.imread('image/image.jpg')
resized_image = cv2.resize(img, (0,0), fx= 0.2, fy=0.2, interpolation=cv2.INTER_AREA)

#drawing raw circle points
for pt in circle_positions:
    cv2.circle(resized_image, pt, 15, (255, 0, 0),-1)
    # predicting trajectory
    predicted = kf.predict(pt[0], pt[1])
    print(predicted)
    cv2.circle(resized_image, predicted, 15, (0, 255, 0), 5)
#predicting next trajectory
# predicted1 = kf.predict(predicted[0], predicted[1])
# cv2.circle(resized_image, predicted1, 15, (0, 255, 0), 5)

# predicting next 10 trajectories
for i in range(10):
    predicted = kf.predict(predicted[0], predicted[1])
    cv2.circle(resized_image, predicted, 15, (0, 255, 0), 5)


cv2.imshow('Image', resized_image)
cv2.waitKey(0)

