import cv2
import numpy as np
from requests import get


# https://github.com/murtazahassan/Learn-OpenCV-in-3-hours/blob/master/chapter6.py
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def get_background_image(webcam):
    width  = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

    image = cv2.imdecode(
        np.frombuffer(
            get(f'https://loremflickr.com/{width}/{height}/star-trek,spock').content,
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    return image


def get_contours(img):
    ret_img = np.zeros_like(img)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest_area = 0
    biggest_contour = None
    x, y, w, h = 0, 0, 0, 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 500:
            continue

        if area > biggest_area:
            biggest_contour = contour

            approx = cv2.approxPolyDP(contour, .03 * cv2.arcLength(contour, True), True)
            x, y, w, h = cv2.boundingRect(approx)

    cv2.drawContours(ret_img, biggest_contour, -1, (255, 255, 255), cv2.FILLED)
    cv2.rectangle(ret_img, (x, y), (x + w, y + h), (128, 255, 255), 2)

    return ret_img
