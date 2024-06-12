import cv2


cap = cv2.VideoCapture(0)

if (cap.isOpened()==False):
    print("Unable to read the video")

frame_width = int(cap.get(3))

frame_height = int(cap.get(4))

out = cv2.VideoWriter('Output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        #resize_frame = cv2.resize(frame, (0,0), fx = 0.2, fy = 0.2, interpolation = cv2.INTER_AREA)
        out.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF==ord('1'):
            break
    else:
        break
out.release()
cap.release()
cv2.destroyAllWindows()