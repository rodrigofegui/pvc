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
    return 'medida: x.xx xx'
