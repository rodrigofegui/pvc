from os import walk as walk_os
from os.path import exists

import cv2 as cv
import numpy as np
from utils.depth_map import save_depth_map
from utils.disparity_map import (save_disp_map, basic_disp_map, get_diff_percent, linear_search_disp_map,
                                 slow_adaptative_windowing_disp_map, windowing_disp_map, x_correlation_disp_map)
from utils.utils import get_pfm_image, get_resize_shape, normalize_map, parse_calib_file, pre_processing
from utils.variables import (ERROR_THRESHOLD, MAX_DISP_FILTER_SZ, MEDIAN_FLT_SZ, MIN_DISP_FILTER_SZ, RESULT_DIR,
                             SHOULD_RESIZE, WORKDIR_C1)

print('Challenge 1: Disparity and Depth maps\n')

for c_dir, next_dir, _ in walk_os(WORKDIR_C1):
    if not exists(f'{c_dir}/calib.txt'):
        continue

    workdir = c_dir.replace(WORKDIR_C1, '')

    calib = parse_calib_file(f'{c_dir}/calib.txt')

    img_left = cv.imread(f'{c_dir}/im0.png', cv.IMREAD_UNCHANGED)
    img_right = cv.imread(f'{c_dir}/im1.png', cv.IMREAD_UNCHANGED)

    lookup_size = calib['disp_n']
    resize_ratio = 1

    disp_map_gt = get_pfm_image(f'{c_dir}/disp0.pfm')

    if SHOULD_RESIZE:
        height, width = get_resize_shape(calib['shape'])
        resize_ratio = height / calib['shape'][0]
        resize_step = int(1 / resize_ratio)
        print(f'resize_ratio: {resize_ratio}')
        print(f'resize_step: {resize_step}')

        img_left = img_left[::resize_step, ::resize_step]
        img_right = img_right[::resize_step, ::resize_step]

        lookup_size = int(lookup_size * resize_ratio)
        lookup_size += 16 - (lookup_size % 16)

        disp_map_gt = disp_map_gt[::resize_step, ::resize_step]
        disp_map_gt *= resize_ratio

    disp_map_gt = np.array(disp_map_gt.reshape(disp_map_gt.shape[:-1]))
    eq_disp_map_gt, max_eq_disp_map_gt = normalize_map(disp_map_gt)

    gray_left = pre_processing(img_left, 1.5, 5)
    gray_right = pre_processing(img_right, 1.5, 5)

    methods = [
        (basic_disp_map, 'OpenCV'),
        (linear_search_disp_map, 'Busca linear'),
        (x_correlation_disp_map, 'Correlação cruzada'),
        (windowing_disp_map, 'Janela deslizante'),
        (slow_adaptative_windowing_disp_map, 'Janela deslizante adaptativa'),
    ]

    with open(f'{RESULT_DIR}/{workdir}/disp_map/analyses.csv', 'a') as out_file:
        out_file.write('method;block_size;error;eq_error\n')

        for method, label in methods:
            for block_sz in range(MIN_DISP_FILTER_SZ, MAX_DISP_FILTER_SZ, 2):
                print(f'method: {label} | block_sz: {block_sz}')
                out_file.write(f'{label};{block_sz};')
                disp_map = method(gray_left, gray_right, block_sz, lookup_size)

                try:
                    disp_map = cv.medianBlur(disp_map, MEDIAN_FLT_SZ)
                except:
                    pass

                perc_errors, diff = get_diff_percent(disp_map, disp_map_gt, ERROR_THRESHOLD)
                out_file.write(f'{str(perc_errors).replace(".", ",")};')
                print(f'\terrors: {perc_errors}', end=' | ')

                eq_disp_map, _ = normalize_map(disp_map)
                perc_errors, diff = get_diff_percent(eq_disp_map, eq_disp_map_gt, ERROR_THRESHOLD)
                out_file.write(f'{str(perc_errors).replace(".", ",")}\n')
                print(f'norm errors: {perc_errors}')

                save_disp_map(
                    disp_map, label,
                    f'{RESULT_DIR}/{workdir}/disp_map/{method.__name__}_bl{block_sz}.png'
                )
                save_disp_map(
                    eq_disp_map, label,
                    f'{RESULT_DIR}/{workdir}/disp_map/eq_{method.__name__}_bl{block_sz}.png'
                )

                save_depth_map(
                    calib, disp_map, label,
                    f'{RESULT_DIR}/{workdir}/depth_map/{method.__name__}_bl{block_sz}.png'
                )
                save_depth_map(
                    calib, eq_disp_map, label,
                    f'{RESULT_DIR}/{workdir}/depth_map/eq_{method.__name__}_eq_bl{block_sz}.png'
                )

print('\nEnding: Challenge 1')
print('-' * 50)
cv.destroyAllWindows()
