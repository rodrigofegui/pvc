from os import cpu_count
from threading import Thread

import cv2 as cv
import numpy as np

from .variables import DISP_BLOCK_SEARCH, DISP_FILTER_RATIO, IMG_LOWEST_DIMENSION, MIN_DISP_FILTER_SZ


def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# https://colab.research.google.com/drive/1Zru_-zzbtylgitbwxbi0eDBNhwr8qYl6#scrollTo=R7GoWaxG2RpX
def roll_2D(a,      # ND array
         b,      # rolling 2D window array
         dx=1,   # horizontal step, abscissa, number of columns
         dy=1):  # vertical step, ordinate, number of rows
    shape = a.shape[:-2] + \
            ((a.shape[-2] - b.shape[-2]) // dy + 1,) + \
            ((a.shape[-1] - b.shape[-1]) // dx + 1,) + \
            b.shape  # sausage-like shape with 2D cross-section
    strides = a.strides[:-2] + \
              (a.strides[-2] * dy,) + \
              (a.strides[-1] * dx,) + \
              a.strides[-2:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def closest_idx(array, value):
    array = np.asarray(array)

    return np.argmin(np.abs(array - value), axis=None)


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
    valid_versions = ['default', 'ajust_window', 'x_correlation', 'linear_search']

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
    disp_map = np.zeros_like(img_left)
    block_sz = int(block_sz / 2)
    # matched = np.zeros((y_range[1], img_left.shape[1]))

    for y in range(img_left.shape[0]):
        print(f'linha: {y}')
        for x_l in range(img_left.shape[1] - 1, -1, -1):
            y_min, y_max = max(0, y - block_sz), min(y + block_sz + 1, img_left.shape[0])
            x_min, x_max = max(0, x_l - block_sz), min(x_l + block_sz + 1, img_left.shape[1])

            x_match = x_l

            ref = img_left[y_min:y_max, x_min:x_max]

            try:
                search = roll_2D(img_right[y:y + ref.shape[0], :x_max], ref)

                # print(f'\tsearch: {search.size}')
                if search.size:
                    search = search[0] - ref
                    x_match = int(closest_idx(search, 0) / ref.size)
            except Exception as e:
                print('error', e)
            finally:
                disp_map[y, x_l] = (x_l - x_match) or 255
        # if y >= 150:
        #     break

    return disp_map

def _get_disp_map_x_correlation(img_left:np.ndarray, img_right:np.ndarray, block_sz:int = MIN_DISP_FILTER_SZ) -> np.ndarray:
    disp_map = np.zeros_like(img_left)

    padding = int(block_sz / 2)

    img_left = np.pad(
        img_left, [(padding, padding), (padding, padding)], mode='constant', constant_values=0
    ).astype(np.int32)
    img_right = np.pad(
        img_right, [(padding, padding), (padding, padding)], mode='constant', constant_values=0
    ).astype(np.int32)

    for unpadded_y, y in enumerate(range(padding, img_left.shape[0] - padding)):
        print(f'linha: {unpadded_y}')

        for unpadded_x, x_l in enumerate(range(padding, img_left.shape[1] - padding - 1)):
            ref = img_left[y, x_l:min(x_l + block_sz if x_l > block_sz else x_l + 1, img_left.shape[1])]

            search = rolling_window(img_right[y, :x_l or 1], ref.size)
            x_correlation = np.dot(search, ref) / (np.linalg.norm(search, axis=1) * np.linalg.norm(ref))

            x_match = x_correlation.argmax()

            disp_map[unpadded_y, unpadded_x] = x_l - x_match

    return disp_map

def _get_disp_map_linear_search(img_left:np.ndarray, img_right:np.ndarray, block_sz:int = MIN_DISP_FILTER_SZ) -> np.ndarray:
    disp_map = np.zeros_like(img_left)

    for y in range(img_left.shape[0]):
        print(f'linha: {y}')
        for x in range(img_left.shape[1]):
            ref = img_left[y, x:min(x + block_sz if x > block_sz else x + 1, img_left.shape[1])]
            search = img_right[y, :x or 1]
            # print(f'ref: {ref.shape}\n{ref}')
            # print(f'search: {search.shape}\n{search}')

            search = rolling_window(search, ref.size) - ref

            x_match = int(closest_idx(search, 0) / ref.size)

            disp_map[y,x] = (x - x_match) or 255

    return disp_map
