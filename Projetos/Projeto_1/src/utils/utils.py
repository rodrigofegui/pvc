from os import cpu_count
from threading import Thread

import cv2 as cv
import numpy as np

from .variables import DISP_BLOCK_SEARCH, DISP_FILTER_RATIO, IMG_LOWEST_DIMENSION, MIN_DISP_FILTER_SZ


def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

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

def _get_disp_map_x_correlation(img_left:np.ndarray, img_right:np.ndarray, block_sz:int = MIN_DISP_FILTER_SZ) -> np.ndarray:
    padding = int(block_sz / 2)

    disp_map = np.zeros_like(img_left)

    img_left = np.pad(
        img_left, [(padding, padding), (padding, padding)], mode='constant', constant_values=0
    ).astype(np.int32)
    img_right = np.pad(
        img_right, [(padding, padding), (padding, padding)], mode='constant', constant_values=0
    ).astype(np.int32)

    for unpadded_y, y in enumerate(range(padding, img_left.shape[0] - padding)):
        print(f'linha: {y}')
        matched_already = np.zeros(img_left.shape[1])

        for unpadded_x, x_l in enumerate(range(img_left.shape[1] - padding - 1, padding - 1, -1)):
            print(f'\tcoluna: {x_l}')
            # ref = img_left[y, x_l - padding:x_l + padding + 1]
            # norm_ref = np.linalg.norm(ref)

            # original_x_correlation = np.sum(np.dot(ref, ref)) / (norm_ref ** 2)
            # correlation_match, x_match = float('inf'), x_l

            # for x_r in range(x_l, padding - 1, -1):
            #     # if matched_already[x_r]:
            #     #     continue
            #     search = img_right[y, x_r - padding:x_r + padding + 1]

            #     c_x_correlation = np.sum(np.dot(ref, search))
            #     c_x_correlation /= norm_ref * np.linalg.norm(search)

            #     diff_x_correlation = (original_x_correlation - c_x_correlation) ** 2

            #     if diff_x_correlation < correlation_match:
            #         correlation_match = diff_x_correlation
            #         x_match = x_r


            # disp_map[unpadded_y, unpadded_x] = (x_l - x_match) or 255
            # matched_already[x_match] = 1

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
