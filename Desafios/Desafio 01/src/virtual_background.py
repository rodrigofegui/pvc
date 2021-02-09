import cv2
import numpy as np
from utils import stackImages


def background_subtraction(webcam, background_img):
    _, key_frame = webcam.read()
    key_frame = cv2.cvtColor(key_frame, cv2.COLOR_BGR2GRAY)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        _, img = webcam.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        diff_img = cv2.subtract(img_gray, key_frame)
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

        masked_img = cv2.bitwise_and(img, img, mask=mask)
        indexes = np.where(masked_img == 0)
        masked_img[indexes] = background_img[indexes]

        stacked_img = stackImages(.5, ([img, background_img, masked_img]))
        cv2.imshow("Imagens", stacked_img)
