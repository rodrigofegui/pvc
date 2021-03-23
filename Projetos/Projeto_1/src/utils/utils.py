from os import cpu_count
from threading import Thread

import cv2 as cv
import numpy as np

from .variables import DISP_BLOCK_SEARCH, DISP_FILTER_RATIO, IMG_LOWEST_DIMENSION, MIN_DISP_FILTER_SZ


def parse_calib_file(file_name):
    with open(file_name, encoding='UTF-8') as file:
        lines = file.readlines()

    lines = [ln.split('=')[1].rstrip('\n') for ln in lines]

    return {
        'intrinsic_0': np.array([*_parse_intrinsic(lines[0])]).astype(float),
        'intrinsic_1': np.array([*_parse_intrinsic(lines[1])]).astype(float),
        'cx_diff': float(lines[2]),
        'baseline': float(lines[3]),
        'shape': (int(lines[5]), int(lines[4])),
        'disp_n': int(lines[6]),
        'is_int': True if int(lines[7]) else False,
        'disp_min': int(lines[8]),
        'disp_max': int(lines[9]),
        'disp_y_avg': float(lines[10]),
        'disp_y_max': int(lines[11]),
    }

def get_resize_shape(original_shape: tuple) -> tuple:
    highest = 'width' if original_shape[0] < original_shape[1] else 'height'

    if highest == 'width':
        return int((IMG_LOWEST_DIMENSION * original_shape[1]) / original_shape[0]), IMG_LOWEST_DIMENSION

    return IMG_LOWEST_DIMENSION, int((IMG_LOWEST_DIMENSION * original_shape[0]) / original_shape[1])

# def get_disp_map(img_left:np.ndarray, img_right:np.ndarray, use_default: bool = False) -> np.ndarray:
def get_disp_map(img_left:np.ndarray, img_right:np.ndarray, version:str = 'ajust_window') -> np.ndarray:
    """version = 'default' | 'ajust_window' """
    valid_versions = ['default', 'ajust_window']

    if version not in valid_versions:
        return np.zeros((*img_left.shape[:-1], 1))

    img_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    img_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    filter_sz = int(max(MIN_DISP_FILTER_SZ, min(*img_left.shape) / DISP_FILTER_RATIO))
    print(f'filter: {filter_sz}')

    return eval(f'_get_disp_map_{version}')(img_left, img_right, filter_sz)

def _parse_intrinsic(raw_line):
    return [
        list(map(float, row.strip().split()))
        for row in raw_line.replace('[', '').replace(']', '').split(';')
    ]

def _get_disp_map_default(img_left:np.ndarray, img_right:np.ndarray, block_sz:int = MIN_DISP_FILTER_SZ) -> np.ndarray:
    return cv.StereoBM_create(
        # numDisparities=img_left.shape[1] - (img_left.shape[1] % 16),
        numDisparities=DISP_BLOCK_SEARCH,
        blockSize=block_sz
    ).compute(img_left, img_right)

def _get_disp_map_ajust_window(img_left:np.ndarray, img_right:np.ndarray, block_sz:int = MIN_DISP_FILTER_SZ) -> np.ndarray:
    img_left, img_right = img_left.astype(np.int32), img_right.astype(np.int32)

    disp_map = np.zeros_like(img_left)

    thread_points = list(range(0, img_left.shape[0] + 1, int(img_left.shape[0] / cpu_count())))
    thread_ranges = [(val, thread_points[ind +1]) for ind, val in enumerate(thread_points[:-1])]

    threads = []
    for rng in thread_ranges:
        t = Thread(
            target=_get_disp_map_ajust_window_esp,
            args=(img_left, img_right, block_sz, rng, disp_map)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return disp_map

def _get_disp_map_ajust_window_esp(
    img_left:np.ndarray, img_right:np.ndarray,
    block_sz:int = MIN_DISP_FILTER_SZ, y_range: tuple = (0, 0),
    result: np.ndarray = None
) -> np.ndarray:
    block_sz = int(block_sz / 2)
    matched = np.zeros((y_range[1], img_left.shape[1]))

    for ind, y in enumerate(range(*y_range)):
        print(f'linha: {y}')
        for x_l in range(img_left.shape[1] - 1, -1, -1):
            sum_match, x_match = float('inf'), x_l

            for x_r in range(np.argmax(matched[y]) or x_l, -1, -1):
                try:
                    x_min, y_min = max(0, x_r - block_sz), max(0, y - block_sz)
                    x_max, y_max = min(x_r + block_sz + 1, img_left.shape[1]), min(y + block_sz + 1, img_left.shape[0])

                    # Shape match ajustment
                    x_l_max = min(x_l + x_max - x_r, img_left.shape[1])
                    x_max = min(x_r + x_l_max - x_l, img_left.shape[1])

                    search = img_right[y_min:y_max, x_min:x_max]
                    ref = img_left[y_min:y_max, x_l + x_min - x_r:x_l_max]
                    diff = abs(ref - search)
                    cmp_sum = np.sum(diff)
                except:
                    print(f'\nRIGHT [{y}, {x_r}]: x_min: {x_min} | x_max: {x_max} | y_min: {y_min} | y_max: {y_max}')
                    print(f'LEFT [{y}, {x_l}]: x_min: {x_l + x_min - x_r} | x_max: {x_l_max} | y_min: {y_min} | y_max: {y_max}')
                    print(f'ref[{y}, {x_l}]: {ref.shape}\n{ref}')
                    print(f'search[{y}, {x_r}]: {search.shape}\n{search}')

                if cmp_sum < sum_match:
                    sum_match = cmp_sum
                    x_match = x_r

            result[y, x_l] = (x_l - x_match) or 255
            matched[y, x_match] = 1

    return result
