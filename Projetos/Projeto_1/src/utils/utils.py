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
    img_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    img_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    if use_default:
        return cv.StereoBM_create(
            numDisparities=DISP_BLOCK_SEARCH,
            blockSize=5
        ).compute(img_left, img_right)

    filter_sz = int(max(MIN_DISP_FILTER_SZ, min(*img_left.shape) / DISP_FILTER_RATIO))
    padding = filter_sz // 2
    padding += 0 if padding % 2 == 0 else 1
    padding = 1
    print('filter', filter_sz, padding)

    disp_map = np.zeros_like(img_left)

    img_left = np.pad(
        img_left, [(padding, padding), (padding, padding)], mode='constant', constant_values=0
    ).astype(np.int32)
    img_right = np.pad(
        img_right, [(padding, padding), (padding, padding)], mode='constant', constant_values=0
    ).astype(np.int32)

    for unpadding_y, y in enumerate(range(padding, img_left.shape[0] - padding)):
    # for unpadding_y, y in enumerate(range(50, img_left.shape[0] - padding)):
        print(f'linha: {y}')
        for unpadding_x, x in enumerate(range(padding, img_left.shape[1] - padding)):
            match_sum, match_x = float('inf'), x
            ref = img_left[y - padding:y + padding + 1, x - padding:x + padding + 1]
            # print(f'ref[{y}, {x}]:\n{ref}')

            for xs in range(x, max(padding, x - DISP_BLOCK_SEARCH) - 1, x + 1):
                search = img_right[y - padding:y + padding + 1, xs - padding:xs + padding + 1]
                diff = abs(ref - search)
                # print(f'search[{y}, {x}]:\n{search}')
                # print(f'diff: {np.sum(diff)} || {match_sum}')
                cmp_sum = np.sum(diff)

                if cmp_sum < match_sum:
                    # print(f'\t{x} -> {xs}')
                    match_sum = cmp_sum
                    match_x = xs

            # input('waiting...')

            disp_map[unpadding_y, unpadding_x] = (x - match_x) or 255
            # print()
            # print(f'disp[{unpadding_y}, {unpadding_x}]: {disp_map[unpadding_y, unpadding_x]}\n')

        # break
        # if unpadding_y >= 150:
        #     break

    return disp_map

def _parse_intrinsic(raw_line):
    return [
        list(map(float, row.strip().split()))
        for row in raw_line.replace('[', '').replace(']', '').split(';')
    ]
