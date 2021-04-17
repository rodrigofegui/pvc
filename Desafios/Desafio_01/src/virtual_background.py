import cv2
import numpy as np
from utils import get_contours


def background_subtraction(cur_frame, fst_frame):
    fst_frame = cv2.cvtColor(fst_frame, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

    diff_img = cv2.subtract(img_gray, fst_frame)
    diff_mean = np.mean(diff_img)

    mask_weight = 1
    if diff_mean <= 70:
        mask_weight = .5
    elif diff_mean <= 85:
        mask_weight = 1.1
    elif diff_mean <= 170:
        mask_weight = .75

    _, mask = cv2.threshold(diff_img, int(diff_mean * mask_weight), 255, cv2.THRESH_BINARY_INV)
    mask = cv2.erode(mask, np.ones((3,3)), iterations=2)
    mask = cv2.dilate(mask, np.ones((3,3)), iterations=3)
    mask = cv2.erode(mask, np.ones((3,3)), iterations=1)

    return mask

def face_detection(cur_frame, _):
    faceCascade= cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')

    gray_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_frame)

    faces = faceCascade.detectMultiScale(gray_frame, 1.1, 4)

    for x, y, w, h in faces:
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0), -1)

    return mask

def contour_detection(cur_frame, _):
    contour_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    canny_val = np.mean(contour_frame) * 1.2

    contour_frame = cv2.GaussianBlur(contour_frame, (5, 5), 1)
    contour_frame = cv2.Canny(contour_frame, canny_val, canny_val)

    kernel = np.ones((5,5))
    contour_frame = cv2.dilate(contour_frame, kernel, iterations=5)
    contour_frame = cv2.erode(contour_frame, kernel, iterations=4)

    contour_frame = get_contours(contour_frame)
    return contour_frame
