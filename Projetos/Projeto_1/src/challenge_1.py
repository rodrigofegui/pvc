from src.utils.utils import cmp_gaussian_blur, cmp_median_blur
import cv2 as cv
import numpy as np
from utils.disparity_map import basic_disp_map, linear_search_disp_map, windowing_disp_map, x_correlation_disp_map
from utils.utils import get_resize_shape, parse_calib_file, normalize_map
from utils.variables import RESULT_DIR, WORKDIRS
from utils.depth_map import get_depth_map
from glob import glob

print('Challenge 1: Disparities and deepness\n')

for workdir in WORKDIRS:
    calib = parse_calib_file(f'{workdir}/calib.txt')
    print(f'calib: {calib}\n')
    height, width = get_resize_shape(calib['shape'])

    img_left = cv.imread(f'{workdir}/im0.png', cv.IMREAD_UNCHANGED)
    img_left = cv.resize(img_left, (height, width), interpolation=cv.INTER_AREA)

    img_right = cv.imread(f'{workdir}/im1.png', cv.IMREAD_UNCHANGED)
    img_right = cv.resize(img_right, (height, width), interpolation=cv.INTER_AREA)

    resize_ratio = img_left.shape[1] / calib['shape'][1]
    lookup_size = int(calib['disp_max'] * resize_ratio)
    lookup_size += 16 - (lookup_size % 16)

    disp_map_gt = cv.imread(f'{workdir}/disp0-sd.pfm', cv.IMREAD_UNCHANGED)
    disp_map_gt = cv.resize(disp_map_gt, (height, width), interpolation=cv.INTER_AREA)
    disp_map_gt *= resize_ratio
    disp_map_gt[disp_map_gt == float('inf')] = lookup_size
    disp_map_gt, disp_map_gt_max = normalize_map(disp_map_gt)

    print(f'img_left.shape: {img_left.shape}')
    print(f'img_right.shape: {img_right.shape}')
    print(f'disp_map_gt.shape: {disp_map_gt.shape} | max: {disp_map_gt_max}')
    print(f'resize_ratio: {resize_ratio}')
    print()

    gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    methods = [
        # basic_disp_map,
        # linear_search_disp_map,
        # windowing_disp_map,
        # x_correlation_disp_map,
    ]

    for method in methods:
        for block_sz in range(5, 17, 2):
        # for block_sz in range(1, 3, 2):
            print(f'method: {method.__name__} | block_sz: {block_sz}')
            disp_map = method(gray_left, gray_right, block_sz, lookup_size)

            print(f'\n\tinvalids: {np.where(disp_map == -1)}')

            file_name = f'{RESULT_DIR}/{method.__name__}_bl{block_sz}'
            cv.imwrite(f'{file_name}.png', disp_map)

            cmp_gaussian_blur(disp_map, disp_map_gt, file_name)

            cmp_median_blur(disp_map_gt, f'{RESULT_DIR}/*.png')


    # for disp_map_name in glob(f'{RESULT_DIR}/Jadeplant-perfect/disp_map/*_bl*.png'):
    #     print(f'ajusting {disp_map_name}')

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
