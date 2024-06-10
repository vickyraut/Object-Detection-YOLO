import cv2
from ultralytics import YOLO
import math

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('../videos/video1.mp4')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

if (cap.isOpened() == False):
    print("Error in Reading the Video")

count = 0
center_point_previous_frame=[]
tracking_objects = {}
tracking_id = 0

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

output = cv2.VideoWriter('output.avi', fourcc, 10, (frame_width, frame_height))

while (cap.isOpened()):
    ret, frame = cap.read()
    center_points_current_frame = []
    count += 1
    if ret:
        # object detections uing yolov8 frame by frame
        # stream = True, if it will be using generators, it is more efficient method
        result = model(frame, stream=True)

        for r in result:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                print(x1, y1, x2, y2)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print("Frame N", count, "", x1, y1, x2, y2)

                # Calculating Center points
                cx = int((x2+x1)/2)
                cy = int((y2+y1)/2)
                center_points_current_frame.append((cx, cy))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'

                # This function calculates the size of the text string label when rendered.
                text_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

                # This line calculates the bottom-right corner of the rectangle that will serve as the background for the text label.
                c2 = x1 + text_size[0], y1 - text_size[1] - 3

                # This function draws a filled rectangle on the frame to serve as the background for the text label.
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1, cv2.LINE_AA)

                # This function renders the text string label on the frame.
                cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                # drawing circle on center points
            for pt in center_points_current_frame:
                for pt2 in center_point_previous_frame:
                    distance = math.hypot(pt2[0]-pt[0], pt2[1]-pt[1])
                    if distance<10:
                        tracking_objects[tracking_id] = pt
                        tracking_id+=1
                for object_id, pt in tracking_objects.items():
                    cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                    cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (255, 0, 0), 3)
                cv2.circle(frame, pt, 5, (0,0,255), -1)

        resize_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        output.write(frame)

        print("Center Points of the Current Frame")
        print(center_points_current_frame)
        print("Center Points of the Previous Frame")
        print(center_point_previous_frame)

        center_point_previous_frame = center_points_current_frame.copy()
        cv2.imshow("Frame", resize_frame)

        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

output.release()
cap.release()
cv2.destroyAllWindows()
