import numpy as np
import cv2

img = cv2.imread('tallinn_road_img.jpg')

def extract_white_and_yellow(img):
    # white color mask
    lower = np.uint8([152, 152, 152])
    upper = np.uint8([225, 225, 225])
    white_mask = cv2.inRange(img, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([200, 200, 200])
    yellow_mask = cv2.inRange(img, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(img, img, mask = mask)

    return masked

def convert_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

masked = extract_white_and_yellow(img)
grayscaled = convert_gray_scale(masked)

cv2.imshow('img', grayscaled)

k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
