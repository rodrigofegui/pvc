from datetime import datetime

import cv2 as cv
import numpy as np

from .variables import DISP_BLOCK_SEARCH, DISP_FILTER_RATIO, IMG_LOWEST_DIMENSION, MIN_DISP_FILTER_SZ, RESULT_DIR


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

def get_disp_map(img_left:np.ndarray, img_right:np.ndarray, use_default: bool = False) -> np.ndarray:
    img_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY).astype(np.int32)
    img_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY).astype(np.int32)

    if use_default:
        return cv.StereoBM_create(
            numDisparities=DISP_BLOCK_SEARCH,
            blockSize=5
        ).compute(img_left, img_right)

    # filter_sz = int(max(MIN_DISP_FILTER_SZ, min(*img_left.shape) / DISP_FILTER_RATIO))
    # padding = filter_sz // 2
    # padding += 0 if padding % 2 == 0 else 1
    filter_sz = None
    padding = 1
    print('filter', filter_sz, padding)

    disp_map = np.zeros_like(img_left)
    matched = np.zeros_like(img_left)

    for y in range(img_left.shape[0]):
    # for unpadding_y, y in enumerate(range(50, img_left.shape[0] - padding)):
        print(f'linha: {y}')
        for x_l in range(img_left.shape[1] - 1, -1, -1):
            # print(f'coluna: {x_l}')
            sum_match, x_match = float('inf'), x_l
            # print(f'ref[{y}, {x}]:\n{ref}')

            for x_r in range(np.argmax(matched[y]) or x_l, -1, -1):
                x_min, y_min = max(0, x_r - padding), max(0, y - padding)
                x_max, y_max = min(x_r + padding + 1, img_left.shape[1]), min(y + padding + 1, img_left.shape[0])

                # Shape match ajustment
                x_l_max = min(x_l + x_max - x_r, img_left.shape[1])
                x_max = min(x_r + x_l_max - x_l, img_left.shape[1])

                search = img_right[y_min:y_max, x_min:x_max]
                ref = img_left[y_min:y_max, x_l + x_min - x_r:x_l_max]
                try:
                    diff = abs(ref - search)
                    cmp_sum = np.sum(diff)
                # print(f'\ndiff: {cmp_sum} {diff.shape}\n{diff}')
                # print(f'search[{y}, {x}]:\n{search}')
                # print(f'diff: {np.sum(diff)} || {match_sum}')
                except:
                    print(f'\nRIGHT [{y}, {x_r}]: x_min: {x_min} | x_max: {x_max} | y_min: {y_min} | y_max: {y_max}')
                    print(f'LEFT [{y}, {x_l}]: x_min: {x_l + x_min - x_r} | x_max: {x_l_max} | y_min: {y_min} | y_max: {y_max}')
                    print(f'ref[{y}, {x_l}]: {ref.shape}\n{ref}')
                    print(f'search[{y}, {x_r}]: {search.shape}\n{search}')

                    print(1/0)

                if cmp_sum < sum_match:
                    # print(f'\t{x_l} -> {x_r}')
                    sum_match = cmp_sum
                    x_match = x_r

            # input('waiting...')

            # print(f'\t{x_l} -> {x_match}')
            disp_map[y, x_l] = (x_l - x_match) or 255
            matched[y, x_match] = 1
            # print()
            # print(f'disp[{unpadding_y}, {unpadding_x}]: {disp_map[unpadding_y, unpadding_x]}\n')

        # break
        # if y >= 32:
        #     break

    return disp_map

def _parse_intrinsic(raw_line):
    return [
        list(map(float, row.strip().split()))
        for row in raw_line.replace('[', '').replace(']', '').split(';')
    ]
