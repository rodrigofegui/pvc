import cv2 as cv
import numpy as np
from utils.depth_map import draw_depth_map
from utils.disparity_map import (basic_disp_map, get_diff_percent, judged_windowing_disp_map, linear_search_disp_map,
                                 windowing_disp_map, x_correlation_disp_map)
from utils.utils import get_pfm_image, get_pgm_image, get_resize_shape, normalize_map, parse_calib_file
from utils.variables import ERROR_THRESHOLD, MEDIAN_FLT_SZ, MIN_DISP_FILTER_SZ, RESULT_DIR, SHOULD_RESIZE, WORKDIRS

print('Challenge 1: Disparities and deepness\n')

for workdir in WORKDIRS:
    calib = parse_calib_file(f'{workdir}/calib.txt')

    img_left = cv.imread(f'{workdir}/im0.png', cv.IMREAD_UNCHANGED)
    img_right = cv.imread(f'{workdir}/im1.png', cv.IMREAD_UNCHANGED)

    lookup_size = calib['disp_n']

    disp_map_gt = get_pfm_image(f'{workdir}/disp0.pfm')

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

    gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    methods = [
        # (basic_disp_map, 'Default OpenCV'),
        # (judged_windowing_disp_map, 'Janela deslizante limitada'),
        # (linear_search_disp_map, 'Busca linear'),
        # (x_correlation_disp_map, 'Correlação cruzada'),
        # (windowing_disp_map, 'Janela deslizante'),
    ]

    for method, label in methods:
        for block_sz in range(MIN_DISP_FILTER_SZ, 17, 2):
            print(f'method: {label} | block_sz: {block_sz}')
            disp_map = method(gray_left, gray_right, block_sz, lookup_size)

            try:
                disp_map = cv.medianBlur(disp_map, MEDIAN_FLT_SZ)
            except:
                pass

            perc_errors, diff = get_diff_percent(disp_map, disp_map_gt, ERROR_THRESHOLD)
            print(f'\terrors: {perc_errors} | min: {np.nanmin(diff)} | max: {np.nanmax(diff)}')

            eq_disp_map, _ = normalize_map(disp_map)
            perc_errors, diff = get_diff_percent(eq_disp_map, eq_disp_map_gt, ERROR_THRESHOLD)
            print(f'\terrors: {perc_errors} | min: {np.nanmin(diff)} | max: {np.nanmax(diff)}')

            cv.imwrite(f'{RESULT_DIR}/{method.__name__}_bl{block_sz}.png', disp_map)
            cv.imwrite(f'{RESULT_DIR}/{method.__name__}_eq_bl{block_sz}.png', eq_disp_map)

            draw_depth_map(
                calib, disp_map, label,
                resize_ratio, f'{RESULT_DIR}/depth_{method.__name__}_bl{block_sz}.png'
            )
            draw_depth_map(
                calib, eq_disp_map, label,
                resize_ratio, f'{RESULT_DIR}/depth_{method.__name__}_eq_bl{block_sz}.png'
            )

print('\nEnding: Challenge 1')
print('-' * 50)
cv.destroyAllWindows()
