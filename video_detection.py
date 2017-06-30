import numpy as np
import cv2

cap = cv2.VideoCapture('tallinn_road_vid.mp4')

print(cap.isOpened())
while True:
    ret, frame = cap.read()
    if ret == True:
        def extract_white_and_yellow(img):
            # white color mask
            lower = np.uint8([170, 170, 170])
            upper = np.uint8([245, 245, 245])
            white_mask = cv2.inRange(img, lower, upper)
            # yellow color mask
            lower = np.uint8([180, 180, 180])
            upper = np.uint8([210, 210, 210])
            yellow_mask = cv2.inRange(img, lower, upper)
            # combine the mask
            mask = cv2.bitwise_or(white_mask, yellow_mask)
            masked = cv2.bitwise_and(img, img, mask = mask)

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
