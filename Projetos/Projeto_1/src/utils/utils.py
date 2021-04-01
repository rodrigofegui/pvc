import cv2 as cv
import numpy as np
from pypfm import PFMLoader
from netpbmfile import NetpbmFile
from .variables import RESULT_DIR, ERROR_THRESHOLD
import re


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

    array = np.asarray(array)[::-1]

    if len(array.shape) < 3:
        return len(array) - np.argmin(np.abs(array - value), axis=None) - 1

    return len(array) - np.argmin(np.sum(np.sum(np.abs(array - value), axis=1), axis=1)) - 1

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
    c_dtype = target.dtype

    target = target.astype(np.float64)

    target_max = np.max(target)

    if target_max == float('inf'):
        target[np.where(target == float('inf')) or np.where(target == float('-inf'))] = np.nan
        target_max = np.nanmax(target)

    target[np.where(target != float('inf')) and np.where(target != float('-inf'))] *= (254 / target_max)
    target[target == np.nan] = 255.

    target = np.round(target, decimals=0)

    return target.astype(c_dtype), target_max

def cmp_gaussian_blur(disp_map: np.ndarray, ground_truth: np.ndarray, method_name, block_sz) -> None:
    from .disparity_map import get_diff_percent

    for kernel in range(5, 17, 2):
        for sigma in range(0, 15):
            # print(f'\tgaussian => kernel: {kernel} | sigma: {sigma}')
            c_disp_map = cv.GaussianBlur(disp_map, (kernel, kernel), sigma)

            errors = get_diff_percent(c_disp_map, ground_truth, ERROR_THRESHOLD)
            # print(f'\t\terrors: {errors}')

            with open(f'{RESULT_DIR}/method_comparison.csv', 'a') as out_file:
                out_file.write(f'{method_name};{block_sz};{kernel};{sigma};{0};{errors}\n')

def cmp_median_blur(disp_map: np.ndarray, ground_truth: np.ndarray, method_name, block_sz) -> None:
    from .disparity_map import get_diff_percent

    for kernel in range(5, 17, 2):
        # print(f'\tmedian => kernel: {kernel}')
        try:
            c_disp_map = cv.medianBlur(disp_map, kernel)

            errors = get_diff_percent(c_disp_map, ground_truth, ERROR_THRESHOLD)
            # print(f'\t\terrors: {errors}')

            with open(f'{RESULT_DIR}/method_comparison.csv', 'a') as out_file:
                out_file.write(f'{method_name};{block_sz};{0};{0};{kernel};{errors}\n')
        except:
            pass

def get_pfm_image(img_name: str) -> np.ndarray:
    # loader = PFMLoader(color=False, compress=True)
    # img = loader.load_pfm(img_name)

    # return np.asarray(img)

    file = open(img_name, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip().decode()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data).copy()
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]

    file.close()

    return data

def get_pgm_image(img_name: str) -> np.ndarray:
    with NetpbmFile(img_name) as pgm:
        img = pgm.asarray()

    return img

def _parse_intrinsic(raw_line):
    return [
        list(map(float, row.strip().split()))
        for row in raw_line.replace('[', '').replace(']', '').split(';')
    ]
