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

def _parse_intrinsic(raw_line):
    return [
        list(map(float, row.strip().split()))
        for row in raw_line.replace('[', '').replace(']', '').split(';')
    ]
