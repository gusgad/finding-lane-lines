import numpy as np
import cv2

cap = cv2.VideoCapture('assets/tallinn_road_vid.mp4')

print(cap.isOpened())
while True:
    ret, frame = cap.read()
    if ret == True:
        def extract_white_and_yellow(img):
            # white color mask
            lower = np.uint8([155, 155, 155])
            upper = np.uint8([240, 240, 240])
            masked = cv2.inRange(img, lower, upper)

            return masked

        masked = extract_white_and_yellow(frame)

        cv2.imshow('img', masked)
        # & 0xFF is required for a 64-bit system
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
