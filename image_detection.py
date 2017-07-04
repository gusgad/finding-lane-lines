import numpy as np
import cv2

img = cv2.imread('assets/tallinn_road_img.jpg')

def extract_white_and_yellow(img):
    # white color mask
    lower = np.uint8([155, 155, 155])
    upper = np.uint8([235, 235, 235])
    mask = cv2.inRange(img, lower, upper)
    # compare to original
    masked = cv2.bitwise_and(img, img, mask = mask)

    return masked

# convert to grayscale
def convert_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# smoothing
def gaussian_blur(img, kernel_size=17):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# detecting edges
def detect_edges(img, low_threshold=30, high_threshold=100):
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
    top_right    = [cols*0.44, rows*0.2]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(img, vertices)

masked = extract_white_and_yellow(img)
grayscaled = convert_gray_scale(masked)
blurred = gaussian_blur(grayscaled)
edges = detect_edges(blurred)
region_of_interest = select_region(edges)

cv2.imshow('img', region_of_interest)

k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
