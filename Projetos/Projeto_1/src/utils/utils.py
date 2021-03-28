import cv2 as cv
import numpy as np


def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# https://colab.research.google.com/drive/1Zru_-zzbtylgitbwxbi0eDBNhwr8qYl6#scrollTo=R7GoWaxG2RpX
def rolling_window_2D(a,      # ND array
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
    if not array.size:
        return -1

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
    from .variables import IMG_LOWEST_DIMENSION

    highest = 'width' if original_shape[0] < original_shape[1] else 'height'

    if highest == 'width':
        return int((IMG_LOWEST_DIMENSION * original_shape[1]) / original_shape[0]), IMG_LOWEST_DIMENSION

    return IMG_LOWEST_DIMENSION, int((IMG_LOWEST_DIMENSION * original_shape[0]) / original_shape[1])

def normalize_map(target: np.ndarray) -> tuple:
    target_max = np.max(target)

    if target_max == float('inf'):
        target_max = target.flatten()
        target_max.sort()
        target_max = target_max[target_max != float('inf')]
        target_max = target_max[-1]

    target = np.round(target * (254 / target_max), decimals=0).astype(np.int32)
    target[target == float('inf')] = 255

    return target, target_max

def cmp_gaussian_blur(disp_map: np.ndarray, ground_truth: np.ndarray, base_file_name: str) -> None:
    max_kernel, max_sigma = 0, 0
    min_sum = float('inf')

    for kernel in range(5, 17, 2):
        for sigma in range(0, 15):
            c_disp_map = cv.GaussianBlur(disp_map, (kernel, kernel), sigma)
            c_disp_map, _ = normalize_map(c_disp_map)

            c_sum = np.sum(np.abs(c_disp_map - ground_truth))

            if c_sum < min_sum:
                min_sum = c_sum
                max_kernel = kernel
                max_sigma = sigma

    c_disp_map = cv.GaussianBlur(disp_map, (max_kernel, max_kernel), max_sigma)
    cv.imwrite(f'{base_file_name}_gau_k{max_kernel}_sg{sigma}_sm{min_sum}.png', c_disp_map)

def cmp_median_blur(ground_truth: np.ndarray, file_names: list) -> None:
    for disp_map_name in file_names:
        print(f'comp {disp_map_name}')

        disp_map = cv.imread(disp_map_name, cv.IMREAD_UNCHANGED)

        min_sum, max_kernel = float('inf'), 0
        for kernel in range(5, 17, 2):
            c_disp_map = cv.medianBlur(disp_map, kernel)
            c_disp_map, _ = normalize_map(disp_map)

            c_sum = np.sum(np.abs(c_disp_map - ground_truth))

            if c_sum < min_sum:
                min_sum = c_sum
                max_kernel = kernel

        c_disp_map = cv.medianBlur(disp_map, max_kernel)
        disp_map_name = disp_map_name[:-4] + f'_med_k{max_kernel}_sm{c_sum}.png'
        cv.imwrite(disp_map_name, c_disp_map)

def _parse_intrinsic(raw_line):
    return [
        list(map(float, row.strip().split()))
        for row in raw_line.replace('[', '').replace(']', '').split(';')
    ]
