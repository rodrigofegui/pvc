import cv2 as cv
import numpy as np


def draw_cursor(image, cursor_lines):
    for start, end in cursor_lines:
        cv.line(image, start, end, (0, 0, 255), 1)

    return image

def draw_clicks(image, points):
    for pt in points:
        cv.circle(image, pt, 3, (255, 0, 0), 3)

    return image

def calculate(matrices, points, dist, size):
    if not ((dist or size) and len(points) == 2):
        return ''

    pt1 = np.array([[*points[0], 1]]).astype(float)
    pt2 = np.array([[*points[1], 1]]).astype(float)

    omega = np.linalg.inv(np.matmul(matrices['intrinsics'], matrices['intrinsics'].T))

    numerator = np.matmul(np.matmul(pt1, omega), pt2.T)
    denominator1 = np.sqrt(np.matmul(np.matmul(pt1, omega), pt1.T))
    denominator2 = np.sqrt(np.matmul(np.matmul(pt2, omega), pt2.T))

    cos_theta = (numerator / (denominator1 * denominator2))[0][0]
    theta = np.arccos(cos_theta)

    if dist:
        quantity = 'size'
        measure = (dist * np.tan(theta / 2)) / 2
    elif size:
        quantity = 'dist'
        measure = size / (2 * np.tan(theta / 2))

    return f'{quantity}: {measure:.2f} cm'
