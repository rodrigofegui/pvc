import os
from inspect import stack

import cv2 as cv
import numpy as np


def stackImages(scale: float = 1., images: None = []) -> np.ndarray:
    """Stacking images in matrix

    Based on: https://github.com/murtazahassan/Learn-OpenCV-in-3-hours/blob/master/chapter6.py

    Args:
    - `scale:float`: Scale factor during composition, default `1` means original size
    - `images:None`: Images to stack

    Returns:
    - Stacked images
    """
    rows, cols = len(images), len(images[0])
    height, width = images[0][0].shape

    if isinstance(images[0], list):
        for x in range (rows):
            for y in range(cols):
                if images[x][y].shape[:2] == images[0][0].shape[:2]:
                    images[x][y] = cv.resize(images[x][y], (0, 0), None, scale, scale)
                else:
                    images[x][y] = cv.resize(images[x][y], (images[0][0].shape[1], images[0][0].shape[0]), None, scale, scale)

                if len(images[x][y].shape) == 2:
                    images[x][y]= cv.cvtColor(images[x][y], cv.COLOR_GRAY2BGR)

        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows

        for x in range(rows):
            hor[x] = np.hstack(images[x])

        ver = np.vstack(hor)
    else:
        for x in range(rows):
            if images[x].shape[:2] == images[0].shape[:2]:
                images[x] = cv.resize(images[x], (0, 0), None, scale, scale)
            else:
                images[x] = cv.resize(images[x], (images[0].shape[1], images[0].shape[0]), None,scale, scale)
            if len(images[x].shape) == 2:
                images[x] = cv.cvtColor(images[x], cv.COLOR_GRAY2BGR)

        hor= np.hstack(images)

        ver = hor

    return ver


def get_absolute_path(relative_path: str = '.', known_path: str = '') -> str:
    '''Considering an known absolute path as start point and a relative path,
    a absolute path is put together

    Args:
    - `relative_path:str`: Relative path from `known_path`
    - `known_path:str`: Reference absolute path

    Returns:
    - Absolute path
    '''
    known_path = known_path or stack()[1].filename

    known_path = os.path.dirname(os.path.realpath(known_path)).split(os.path.sep)
    relative_path = relative_path.split(os.path.sep)

    while relative_path[0] == '..':
        relative_path.pop(0)
        known_path.pop()

    return os.path.sep.join(known_path + relative_path)


def make_sure_path_exists(path_name: str, is_dir: bool = True) -> str:
    if is_dir:
        path_name = os.path.dirname(path_name)

    if not os.path.exists(path_name):
        if is_dir:
            os.makedirs(path_name)
        else:
            os.mknod(path_name)

    return path_name
