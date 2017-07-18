import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
import sys

cap = cv2.VideoCapture(str(sys.argv[1]))
out = cv2.VideoWriter('tallinn.avi', -1, 20.0, (1024,960))

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
            top_left     = [cols*0.45, rows*0.2]
            bottom_right = [cols*0.9, rows*0.95]
            top_right    = [cols*0.44, rows*0.5]
            # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
            vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
            return filter_region(img, vertices)

        def hough_lines(img):
            return cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=300)

        def draw_lines(img, lines, color=[0, 0, 255], thickness=10, make_copy=True):
            # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
            if make_copy:
                img = np.copy(img) # don't want to modify the original
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            return img

        def average_slope_intercept(lines):
            left_lines    = [] # (slope, intercept)
            left_weights  = [] # (length,)
            right_lines   = [] # (slope, intercept)
            right_weights = [] # (length,)

            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x2==x1:
                        continue # ignore a vertical line
                    slope = (y2-y1)/(x2-x1)
                    intercept = y1 - slope*x1
                    length = np.sqrt((y2-y1)**2+(x2-x1)**2)
                    if slope < 0: # y is reversed in image
                        left_lines.append((slope, intercept))
                        left_weights.append((length))
                    else:
                        right_lines.append((slope, intercept))
                        right_weights.append((length))

            # add more weight to longer lines
            left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
            right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None

            return left_lane, right_lane # (slope, intercept), (slope, intercept)

        def make_line_points(y1, y2, line):
            if line is None:
                return None

            slope, intercept = line

            # make sure everything is integer as cv2.line requires it
            x1 = int((y1 - intercept)/slope)
            x2 = int((y2 - intercept)/slope)
            y1 = int(y1)
            y2 = int(y2)

            return ((x1, y1), (x2, y2))

        def lane_lines(img, lines):
            left_lane, right_lane = average_slope_intercept(lines)

            y1 = img.shape[0] # bottom of the image
            y2 = y1*0.6         # slightly lower than the middle

            left_line  = make_line_points(y1, y2, left_lane)
            right_line = make_line_points(y1, y2, right_lane)

            return left_line, right_line


        def draw_lane_lines(img, lines, color=[255, 0, 0], thickness=50):
            # make a separate image to draw lines and combine with the orignal later
            line_image = np.zeros_like(img)
            for line in lines:
                if line is not None:
                    cv2.line(line_image, *line,  color, thickness)
            # image1 * α + image2 * β + λ
            # image1 and image2 must be the same shape.
            return cv2.addWeighted(img, 1.0, line_image, 0.95, 0.0)

        masked = extract_white_and_yellow(frame)
        blurred = gaussian_blur(masked)
        edges = detect_edges(blurred)
        region_of_interest = select_region(edges)
        hough_lines = hough_lines(region_of_interest)
        lines = draw_lines(frame, hough_lines)
        draw_lane_lines(frame, lane_lines(frame, hough_lines))

        out.write(lines)
        cv2.imshow('vid', lines)
        # & 0xFF is required for a 64-bit system
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
