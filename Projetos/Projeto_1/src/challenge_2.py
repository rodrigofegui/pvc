from random import randint

import cv2 as cv
import numpy as np
from numpy.linalg.linalg import norm
from utils.depth_map import draw_depth_map
from utils.disparity_map import basic_disp_map
from utils.utils import (closest_idx, gamma_ajust, get_resize_shape, isolate_action_figure, normalize_map,
                         parse_camera_c2)
from utils.variables import MEDIAN_FLT_SZ, MIN_DISP_FILTER_SZ, RESULT_DIR, WORKDIRS_C2

BEST_MATCH_PERC = .9
MATCHES_PER_PT = 2

print('Challenge 2: Stereo cameras with convergence\n')


# https://github.com/murtazahassan/Learn-OpenCV-in-3-hours/blob/master/chapter6.py
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def vec_2_mat(src: np.ndarray) -> np.ndarray:
    if src.shape != (3, ):
        return np.full((3,3), fill_value=-1)

    return np.array([
        [0, -src[2], src[1]],
        [src[2], 0, -src[0]],
        [-src[1], src[0], 0]
    ])

for workdir in WORKDIRS_C2:

    img_left = cv.imread(f'{workdir}L.jpg', cv.IMREAD_COLOR)

    height, width = get_resize_shape(img_left.shape)

    img_left = cv.resize(img_left, (height, width), img_left, fx=0, fy=0)

    img_right = cv.imread(f'{workdir}R.jpg', cv.IMREAD_COLOR)
    img_right = cv.resize(img_right, (height, width), img_right, fx=0, fy=0)

    camera_L_info = parse_camera_c2(f'{workdir}L.txt')
    camera_R_info = parse_camera_c2(f'{workdir}R.txt')

    print(f'camera_L_info:\n{camera_L_info}')
    print(f'camera_R_info:\n{camera_R_info}')
    # result dir 'o'
    # gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    # gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    # result dir 'i'
    # gray_left = cv.cvtColor(isolate_action_figure(img_left), cv.COLOR_BGR2GRAY)
    # gray_right = cv.cvtColor(isolate_action_figure(img_right), cv.COLOR_BGR2GRAY)

    # result dir 'ie'
    gray_left = cv.equalizeHist(cv.cvtColor(isolate_action_figure(img_left), cv.COLOR_BGR2GRAY))
    gray_right = cv.equalizeHist(cv.cvtColor(isolate_action_figure(img_right), cv.COLOR_BGR2GRAY))

    sift = cv.SIFT_create(contrastThreshold=.007, edgeThreshold=20)

    raw_key_pts_L, descriptors_L = sift.detectAndCompute(gray_left, None)
    raw_key_pts_R, descriptors_R = sift.detectAndCompute(gray_right, None)

    print(f'raw_key_pts_L: {len(raw_key_pts_L)}')

    FLANN_INDEX_KDTREE = 1
    index_params = {
        'algorithm': FLANN_INDEX_KDTREE,
        'trees': 5
    }
    search_params = {'checks': 50}

    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors_L, descriptors_R, k=MATCHES_PER_PT)

    key_pts_L = []
    key_pts_R = []
    # ratio test as per Lowe's paper
    for m_L, m_R in matches:
        if m_L.distance < BEST_MATCH_PERC * m_R.distance:
            key_pts_L.append(raw_key_pts_L[m_L.queryIdx].pt)
            key_pts_R.append(raw_key_pts_R[m_L.trainIdx].pt)

    key_pts_L = np.array(key_pts_L, dtype=np.int32)
    key_pts_R = np.array(key_pts_R, dtype=np.int32)

    fund_matrix, mask = cv.findFundamentalMat(key_pts_L, key_pts_R, cv.FM_LMEDS)
    print(f'opencv F: {fund_matrix.shape}\n{fund_matrix}')

    # # We select only inlier points
    key_pts_L = key_pts_L[mask.ravel() == 1]
    key_pts_R = key_pts_R[mask.ravel() == 1]

    _, h1, h2 = cv.stereoRectifyUncalibrated(key_pts_L, key_pts_R, fund_matrix, gray_left.shape)
    print(f'h1:\n{h1}')
    print(f'h2:\n{h2}')

    ret_img_left = cv.warpPerspective(gray_left, h1, (img_left.shape[1], img_left.shape[0]))
    ret_img_right = cv.warpPerspective(gray_right, h2, (img_right.shape[1], img_right.shape[0]))

    workdir_name = workdir.split("/")[-1]
    for block_sz in range(MIN_DISP_FILTER_SZ, 25, 2):
        for lookup in range(MIN_DISP_FILTER_SZ, 80, 5):
            print(f'block_sz: {block_sz} | lookup: {lookup}')
            disp_map = basic_disp_map(ret_img_left, ret_img_right, block_sz, lookup)

            try:
                disp_map = cv.medianBlur(disp_map, MEDIAN_FLT_SZ)
            except:
                pass

            eq_disp_map, _ = normalize_map(disp_map)
            eq_disp_map = cv.equalizeHist(eq_disp_map.astype('uint8'))

            cv.imwrite(f'{RESULT_DIR}/ie/{workdir_name}_bl{block_sz}_lk{lookup}.png', disp_map)
            cv.imwrite(f'{RESULT_DIR}/ie/{workdir_name}_eq_bl{block_sz}_lk{lookup}.png', eq_disp_map)

    continue

print('\nEnding: Challenge 2')
print('-' * 50)
cv.destroyAllWindows()
