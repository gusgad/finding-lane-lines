import numpy as np
import cv2

img = cv2.imread('assets/tallinn_road_img.jpg')

def extract_white_and_yellow(img):
    # white color mask
    lower = np.uint8([155, 155, 155])
    upper = np.uint8([235, 235, 235])
    mask = cv2.inRange(img, lower, upper)
    # combine the mask
    masked = cv2.bitwise_and(img, img, mask = mask)

    return masked

def convert_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size=15):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

masked = extract_white_and_yellow(img)
grayscaled = convert_gray_scale(masked)
blurred = gaussian_blur(grayscaled)

cv2.imshow('img', blurred)

k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
