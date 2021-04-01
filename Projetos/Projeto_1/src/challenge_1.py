from glob import glob

import cv2 as cv
import numpy as np
from utils.depth_map import calc_depth_map, get_depth_map
from utils.disparity_map import (basic_disp_map, judged_windowing_disp_map, linear_search_disp_map, windowing_disp_map,
                                 x_correlation_disp_map, get_diff_percent)
from utils.utils import cmp_gaussian_blur, cmp_median_blur, get_resize_shape, normalize_map, parse_calib_file, get_pfm_image, get_pgm_image
from utils.variables import RESULT_DIR, WORKDIRS, ERROR_THRESHOLD

print('Challenge 1: Disparities and deepness\n')

for workdir in WORKDIRS:
    calib = parse_calib_file(f'{workdir}/calib.txt')
    print(f'calib: {calib}\n')
    height, width = get_resize_shape(calib['shape'])

    resize_ratio = height / calib['shape'][0]
    resize_step = int(1 / resize_ratio)
    print(f'resize_ratio: {resize_ratio}')
    print(f'resize_step: {resize_step}')

    img_left = cv.imread(f'{workdir}/im0.png', cv.IMREAD_UNCHANGED)
    img_left = img_left[::resize_step, ::resize_step]
    # img_left = cv.resize(img_left, (height, width), interpolation=cv.INTER_AREA)
    print(f'img_left.shape: {img_left.shape}')

    img_right = cv.imread(f'{workdir}/im1.png', cv.IMREAD_UNCHANGED)
    img_right = img_right[::resize_step, ::resize_step]
    print(f'img_right.shape: {img_right.shape}')
    # img_right = cv.resize(img_right, (height, width), interpolation=cv.INTER_AREA)

    lookup_size = calib['disp_n']
    lookup_size = int(lookup_size * resize_ratio)
    lookup_size += 16 - (lookup_size % 16)
    print(f'lookup_size: {lookup_size}')

    # disp_map_gt = get_pgm_image(f'{workdir}/disp0-n.pgm')
    disp_map_gt = np.round(get_pfm_image(f'{workdir}/disp0.pfm'), decimals=0)
    disp_map_gt = disp_map_gt[::resize_step, ::resize_step]
    disp_map_gt *= resize_ratio
    # disp_map_gt[disp_map_gt == float('inf')] = lookup_size
    # disp_map_gt, disp_map_gt_max = normalize_map(disp_map_gt)
    disp_map_gt = disp_map_gt.reshape(disp_map_gt.shape[:-1])
    eq_disp_map_gt, max_eq_disp_map_gt = normalize_map(disp_map_gt)
    # cv.imwrite(f'{RESULT_DIR}/eq_disp_map_gt.png', eq_disp_map_gt)
    print(f'disp_map_gt.shape: {eq_disp_map_gt.shape} | med: {np.nanmean(eq_disp_map_gt)} | max: {max_eq_disp_map_gt}')
    print(f'eq_disp_map_gt:\n{eq_disp_map_gt}')

    print(f'error_threshold: {ERROR_THRESHOLD}')

    gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    methods = [
        basic_disp_map,
        judged_windowing_disp_map,
        linear_search_disp_map,
        x_correlation_disp_map,
        windowing_disp_map,
    ]

    for method in methods:
        for block_sz in range(5, 6, 2):
        # for block_sz in range(5, 7, 2):
            print(f'method: {method.__name__} | block_sz: {block_sz}')
            disp_map = method(gray_left, gray_right, block_sz, lookup_size)
            # eq_disp_map, _ = normalize_map(disp_map)
            print(f'eq_disp_map:\n{disp_map}')
            cv.imwrite(f'{RESULT_DIR}/{method.__name__}.png', disp_map)

            errors = get_diff_percent(disp_map, eq_disp_map_gt, ERROR_THRESHOLD)

            print(f'\terrors: {errors}')

            # with open(f'{RESULT_DIR}/method_comparison.csv', 'a') as out_file:
            #     out_file.write(f'{method.__name__};{block_sz};{0};{0};{0};{errors}\n')

            # cmp_gaussian_blur(disp_map, disp_map_gt, method.__name__, block_sz)

            # cmp_median_blur(disp_map, disp_map_gt, method.__name__, block_sz)


    # calc_depth_map(calib, glob(f'{RESULT_DIR}/Jadeplant-perfect/disp_map/x_correlation_disp_map_bl15_med_k7_sm33658864.png'), f'{RESULT_DIR}')
    #     disp_map = cv.imread(disp_map_name, cv.IMREAD_UNCHANGED)
    #     depth_map = get_depth_map(calib, disp_map)

    #     depth_map_max = np.max(depth_map)

    #     if depth_map_max == float('inf'):
    #         depth_map_max = depth_map.flatten()
    #         depth_map_max.sort()
    #         depth_map_max = depth_map_max[depth_map_max != float('inf')]
    #         depth_map_max = depth_map_max[-1]

    #     depth_map = np.round(depth_map * (254 / depth_map_max), decimals=0).astype(np.int32)
    #     depth_map[depth_map == float('inf')] = 255

    #     # print(f'\tmax: {depth_map_max} -> {np.max(depth_map)}')

    #     depth_map_name = f'{RESULT_DIR}/Jadeplant-perfect/depth_map/de_{disp_map_name.split("/")[-1]}'
    #     cv.imwrite(depth_map_name, depth_map)


print('\nEnding: Challenge 1')
print('-' * 50)
cv.destroyAllWindows()
