from random import randint

from numpy.linalg.linalg import norm
from utils.disparity_map import basic_disp_map
import numpy as np
import cv2 as cv
from utils.variables import WORKDIRS_C2
from utils.utils import parse_camera_c2, get_resize_shape, closest_idx, normalize_map

BEST_MATCH_PERC = .9
MATCHES_PER_PT = 2

print('Challenge 2: Stereo cameras with convergence\n')

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
    gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)

    img_right = cv.imread(f'{workdir}R.jpg', cv.IMREAD_COLOR)
    img_right = cv.resize(img_right, (height, width), img_right, fx=0, fy=0)
    gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    camera_L_info = parse_camera_c2(f'{workdir}L.txt')
    camera_R_info = parse_camera_c2(f'{workdir}R.txt')

    print(f'camera_L_info:\n{camera_L_info}')
    print(f'camera_R_info:\n{camera_R_info}')

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
    print(f'ret_img_left: {ret_img_left.shape}')

    win_size = 3
    min_disp = -1
    num_disp = 110 - min_disp
    num_disp -= num_disp % 16
    stereo_L = cv.StereoSGBM(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=16,
        P1=8*1*win_size^2,
        P2=32*1*win_size^2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv.STEREO_SGBM_MODE_SGBM
    )
    stereo_R = cv.ximgproc.createRightMatcher(stereo_L)
    wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo_L)

    disp_map_L = stereo_L.compute(ret_img_left, ret_img_right)
    disp_map_R = stereo_R.compute(ret_img_left, ret_img_right)
    # disp_map_L = cv.normalize(disp_map_L, disp_map_L, 255, 0, cv.NORM_MINMAX)

    # disp_map_L = basic_disp_map(ret_img_left, ret_img_right, 3, 180)
    # disp_map_R = basic_disp_map(ret_img_right, ret_img_left, 3, 180)
    disp_map_L, _ = normalize_map(disp_map_L)

    cv.imshow('ret_img_left', ret_img_left)
    cv.imshow('ret_img_right', ret_img_right)
    cv.imshow('disp_map_L', disp_map_L)
    # cv.imshow('disp_map_R', disp_map_R)
    # cv.imshow('filtered_disp_map', filtered_disp_map)
    cv.waitKey(100000)

    break

    rnd_idx = randint(0, len(key_pts_L))
    base_L = key_pts_L[rnd_idx]
    base_R = key_pts_R[rnd_idx]
    print(f'base_L: {base_L.shape}\n{base_L}')
    cv.circle(img_left, (base_L[1], base_L[0]), 3, (255, 0, 0), 2)
    cv.circle(img_right, (base_R[1], base_R[0]), 3, (255, 0, 0), 2)
    # print(f'base_kp_L: {base_kp_L} : {np.argmin(np.abs(key_pts_L - base_kp_L))}')

    # # # Find epilines corresponding to points in right image (second image) and
    # # # drawing its lines on left image
    # lines_L = cv.computeCorrespondEpilines(key_pts_R.reshape(-1,1,2), 2, fund_matrix)
    # lines_L = lines_L.reshape(-1,3)
    # # print(f'lines_L: {lines_L.shape}\n{lines_L}')
    # img5,img6 = drawlines(gray_left, gray_right, lines_L, key_pts_L, key_pts_R)
    # # Find epilines corresponding to points in left image (first image) and
    # # drawing its lines on right image
    # lines_R = cv.computeCorrespondEpilines(key_pts_L.reshape(-1,1,2), 1, fund_matrix)
    # lines_R = lines_R.reshape(-1,3)
    # img3,img4 = drawlines(gray_right, gray_left, lines_R, key_pts_R, key_pts_L)
    # cv.imshow('epi left', img5)
    # cv.imshow('epi right', img3)

    # l_line = np.matmul(fund_matrix, base_kp_L)
    # print(f'l_line: {l_line.shape}\n{l_line}')
    # l_line = np.delete(np.insert(l_line, 0, l_line[-1]), -1)
    # print(f'l_line: {l_line.shape}\n{l_line}')
    # rows_r, cols_r = np.indices(gray_right.shape)
    # coord_array = np.ones_likes(gray_right)
    # coord_array[0, :, :] = rows_r
    # coord_array[1, :, :] = cols_r
    # print(f'rows_r: {rows_r.shape}')
    # print(f'cols_r: {cols_r.shape}')
    # print(f'coord_array: {coord_array.shape}')
    # t = coord_array * l_line
    # # l_line = np.matmul(l_line, np.array([60, 40, 1]).T)
    # print(f't: {t.shape}\n\te.g: {t[50][5]}')
    # t = np.sum(t, axis=0)
    # print(f't: {t.shape}\n{t[50]}')

    # patch_squares = np.zeros_like(gray_left)
    # disp_map = np.zeros_like(gray_left)

    # FILTER_SZ = 7
    # FILTER_CENTER = int(np.floor(FILTER_SZ / 2))
    # bfm = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    # # # matches = [m for m in bfm.match(descriptors_L, descriptors_R) if m.distance < 2000]
    # matches = bfm.match(descriptors_L, descriptors_R)
    # matches = sorted(matches, key = lambda x:x.distance)
    # print(f'matches: {len(matches)}')
    # for c_match in matches:
    #     # patch_P = np.full((FILTER_SZ, FILTER_SZ), float('inf'))

    #     pt_L = np.asarray(raw_key_pts_L[c_match.queryIdx].pt, dtype=np.int32)
    #     pt_R = np.asarray(raw_key_pts_R[c_match.trainIdx].pt, dtype=np.int32)

    #     diff = np.abs(pt_L - pt_R)
    #     theta = np.degrees(np.arctan(diff[1] / diff[0]))

    #     # if theta <= 20:
    #     xx, yy = np.mgrid[:FILTER_SZ, :FILTER_SZ]
    #     patch_P = ((xx - FILTER_CENTER) ** 2 + (yy - FILTER_CENTER) ** 2).astype(np.float32)
    #     threshold = patch_P[0][FILTER_CENTER]
    #     patch_P[patch_P > threshold] = float('inf')
    #     patch_P[patch_P <= threshold] = np.round(diff[1], decimals=2)


    #     y_min, y_max = max(0, pt_L[1] - FILTER_CENTER), min(pt_L[1] + FILTER_CENTER + 1, patch_squares.shape[0])
    #     x_min, x_max = max(0, pt_L[0] - FILTER_CENTER), min(pt_L[0] + FILTER_CENTER + 1, patch_squares.shape[1])

    #     y_p_min, y_p_max = max(0, FILTER_CENTER - (pt_L[1] - y_min)), min(FILTER_SZ, FILTER_CENTER + (y_max - pt_L[1]))
    #     x_p_min, x_p_max = max(0, FILTER_CENTER - (pt_L[0] - x_min)), min(FILTER_SZ, FILTER_CENTER + (x_max - pt_L[0]))

    #     patch_squares[y_min:y_max, x_min:x_max] = patch_P[y_p_min:y_p_max, x_p_min:x_p_max]

    #     # print(f'pt_L: {pt_L.shape} -> {pt_L}')
    #     # print(f'pt_R: {pt_R.shape} -> {pt_R}')
    #     # print(f'diff: {diff.shape} -> {diff}')
    #     # print(f'[{y_min} -> {y_max}] & [{x_min} -> {x_max}]')
    #     # print(f'theta: {theta}')
    #     # print(f'patch_P: {threshold} -> {patch_P.shape}\n{patch_P}')
    #     # break

    # for y in disp_map.shape[0]:
    #     for x in disp_map.shape[1]:
    #         pass

    # cv.imshow('disp', patch_squares)
    img3 = cv.drawMatches(img_left, raw_key_pts_L, img_right, raw_key_pts_R, matches[rnd_idx], img_right, flags=2)
    cv.imshow('matches', img3)

    # base_idx = choice(range(len(keys_L)))
    # base_key = keys_L[base_idx]
    # base_desc = descriptors_L[base_idx]

    # # match_key = [k for k in keys_R if cv.KeyPoint_overlap(base_key, k)][0]

    # print(f'angle: {base_key.angle}')
    # print(f'octave: {base_key.octave}')
    # print(f'pt: {base_key.pt}')
    # print(f'response: {base_key.response}')
    # print(f'size: {base_key.size}')
    # print('\n' * 2)
    # # print(f'descriptor: {base_desc}')
    # print(f'descriptor match: {descriptors_R[np.where(descriptors_R == base_desc)]}')
    # print(f'base_key in keys_R: {match_key}')
    # print(f'angle: {match_key.angle}')
    # print(f'octave: {match_key.octave}')
    # print(f'pt: {match_key.pt}')
    # print(f'response: {match_key.response}')
    # print(f'size: {match_key.size}')

    # cv.circle(img_left, tuple(map(int, base_L.pt)), 5, (255,0,0), 3)
    # cv.circle(img_right, tuple(map(int, base_R.pt)), 5, (255,0,0), 3)

    # marked_L = cv.drawKeypoints(gray_left, keys_L, img_left)
    # marked_R = cv.drawKeypoints(gray_right, keys_R, img_right)

    cv.imshow('gray_left', img_left)
    cv.imshow('gray_right', img_right)
    cv.waitKey(100000)

print('\nEnding: Challenge 2')
print('-' * 50)
cv.destroyAllWindows()
