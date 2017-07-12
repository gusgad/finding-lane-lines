import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np

cap = cv2.VideoCapture('assets/tallinn_road_vid2.avi')

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

        # smoothing
        def gaussian_blur(img, kernel_size=17):
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # detecting edges
        def detect_edges(img, low_threshold=30, high_threshold=110):
            return cv2.Canny(img, low_threshold, high_threshold)

        def filter_region(img, vertices):
            mask = np.zeros_like(img)
            if len(mask.shape)==2:
                cv2.fillPoly(mask, vertices, 255)
            else:
                cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension
            return cv2.bitwise_and(img, mask)

        def select_region(img):
            # first, define the polygon by vertices
            rows, cols = img.shape[:2]
            bottom_left  = [cols*0.1, rows*0.95]
            top_left     = [cols*0.45, rows*0.25]
            bottom_right = [cols*0.9, rows*0.95]
            top_right    = [cols*0.44, rows*0.25]
            # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
            vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
            return filter_region(img, vertices)

        def hough_lines(img):
            return cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=300)

        def draw_lines(img, lines, color=[0, 0, 255], thickness=2, make_copy=True):
            # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
            if make_copy:
                img = np.copy(img) # don't want to modify the original
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            return img

        masked = extract_white_and_yellow(frame)
        blurred = gaussian_blur(masked)
        edges = detect_edges(blurred)
        region_of_interest = select_region(edges)
        hough_lines = hough_lines(region_of_interest)
        lines = draw_lines(frame, hough_lines)

        cv2.imshow('img', region_of_interest)
        # & 0xFF is required for a 64-bit system
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
