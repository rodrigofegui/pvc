from os import walk as walk_os
from os.path import exists

import cv2 as cv
from utils.disparity_map import save_disp_map, windowing_disp_map
from utils.utils import get_key_pts, get_resize_shape, parse_camera_c2, pre_processing, rectify_corresp_imgs
from utils.variables import MAX_DISP_LOOKUP, MEDIAN_FLT_SZ, MIN_DISP_LOOKUP, RESULT_DIR, SHOULD_RESIZE, WORKDIR_C2

print('Challenge 2: Stereo cameras with convergence\n')

for c_dir, next_dir, _ in walk_os(WORKDIR_C2):
    workdir = c_dir.replace(WORKDIR_C2, '')

    if workdir != 'warrior': continue

    if not exists(f'{c_dir}/{workdir}L.txt'):
        continue

    img_left = cv.imread(f'{c_dir}/{workdir}L.jpg', cv.IMREAD_COLOR)
    img_right = cv.imread(f'{c_dir}/{workdir}R.jpg', cv.IMREAD_COLOR)

    if SHOULD_RESIZE:
        height, width = get_resize_shape(img_left.shape)

        img_left = cv.resize(img_left, (height, width), img_left, fx=0, fy=0)
        img_right = cv.resize(img_right, (height, width), img_right, fx=0, fy=0)
    else:
        img_right = cv.resize(img_right, (img_left.shape[1], img_left.shape[0]), img_right, fx=0, fy=0)

    camera_L_info = parse_camera_c2(f'{c_dir}/{workdir}L.txt')
    camera_R_info = parse_camera_c2(f'{c_dir}/{workdir}R.txt')

    gray_left = pre_processing(img_left, 1.5, 5)
    gray_right = pre_processing(img_right, 1.5, 5)

    key_pts_L, key_pts_R = get_key_pts(gray_left, gray_right)

    ret_img_L, ret_img_R = rectify_corresp_imgs(gray_left, gray_right, key_pts_L, key_pts_R)
    print('ret_img_L', ret_img_L.shape, ret_img_L.dtype)
    print('ret_img_R', ret_img_R.shape, ret_img_R.dtype)

    for lookup in range(MIN_DISP_LOOKUP, MAX_DISP_LOOKUP, 150):
        print(f'block_sz: {15} | lookup: {lookup}')
        disp_map = windowing_disp_map(ret_img_L, ret_img_R, 15, lookup)

        try:
            disp_map = cv.medianBlur(disp_map, MEDIAN_FLT_SZ)
        except:
            pass

        save_disp_map(
            disp_map, 'Janela deslizante',
            f'{RESULT_DIR}/{workdir}/disp_map/lk{lookup}.png'
        )

print('\nEnding: Challenge 2')
print('-' * 50)
cv.destroyAllWindows()
