from re import match

import cv2 as cv
import numpy as np
from netpbmfile import NetpbmFile


def get_pfm_image(img_name: str) -> np.ndarray:
    """Get the .PFM image

    Source: https://gist.github.com/chpatrick/8935738

    Args:
    - `img_name:str`: .PFM image file name

    Returs:
    - PFM Image
    """
    file = open(img_name, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip().decode()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = match(r'^(\d+)\s(\d+)\s$', file.readline().decode())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data).copy()
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]

    file.close()

    return data


def get_pgm_image(img_name: str) -> np.ndarray:
    """Get the .PGM image

    Args:
    - `img_name:str`: : .PGM image file name

    Returns:
    - PGM Image
    """
    with NetpbmFile(img_name) as pgm:
        img = pgm.asarray()

    return img


def parse_calib_file(file_name: str) -> dict:
    """Parse raw calibration file into a dict

    Args:
    - `file_name:str`: Calibration file name

    Returns:
    - Parsed calibration info
    """
    with open(file_name, encoding='UTF-8') as file:
        lines = file.readlines()

    lines = [ln.split('=')[1].rstrip('\n') for ln in lines]

    return {
        'intrinsic_0': _parse_intrinsic(lines[0]),
        'intrinsic_1': _parse_intrinsic(lines[1]),
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


def parse_camera_c2(file_name: str) -> dict:
    """Parse raw challenge #2 calibration file into a dict

    Args:
    - `file_name:str`: Calibration file name

    Returns:
    - Parsed calibration info
    """
    with open(file_name, encoding='UTF-8') as camera_file:
        lines = camera_file.readlines()

    intrinsic_matrix = np.zeros((3, 3), dtype=np.float64)
    extrinsic_matrix = np.zeros((3, 4), dtype=np.float64)

    intrinsic_matrix[2][2] = 1

    reading_rotation = 0

    for line in lines:
        if line.startswith('%'):
            continue

        line = line.rstrip('\n').replace('[', '').replace(']', '').strip().split('=')

        if line[0].startswith('fc'):
            cx, cy = line[1].strip().split('; ')

            intrinsic_matrix[0][0] = float(cx)
            intrinsic_matrix[1][1] = float(cy)

        elif line[0].startswith('cc'):
            cx, cy = line[1].strip().split('; ')

            intrinsic_matrix[0][2] = float(cx)
            intrinsic_matrix[1][2] = float(cy)

        elif line[0].startswith('alpha'):
            skew = line[1].strip().split(';')[0]

            intrinsic_matrix[0][1] = float(skew)

        elif line[0].startswith('R'):
            values = [float(v.strip()) for v in line[1].strip().rstrip(' ;').split(',')]

            extrinsic_matrix[0, :3] = values
            reading_rotation = 1

        elif line[0].startswith('Tc'):
            extrinsic_matrix[:, 3] = np.asarray([float(v.strip()) for v in line[1].strip().split(';')]).reshape((1, 3))

        elif reading_rotation and line[0]:
            values = [v.strip() for v in line[0].strip().rstrip(' ;').split(',')]

            extrinsic_matrix[reading_rotation, :3] = values
            reading_rotation += 1

    return {
        'intrinsic': intrinsic_matrix,
        'extrinsic': extrinsic_matrix,
    }


def pre_processing(img_src: np.ndarray, contrast: float=1.5, brightness: int=20) -> np.ndarray:
    """Common pre processing to ajust original image contrast and brightness then
    equalizing its histogram

    Args:
    - `img_src:np.ndarray`: Raw image
    - `contrast:float`: Contrast ajust, the higher value the whiter image
    - `brightness:int`: Brightness ajust, the higher value the whiter image

    Returns:
    - Ajusted image
    """
    img_dst = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)
    img_dst = np.clip((img_dst * contrast) + brightness, 0, 255).astype(np.uint8)
    img_dst = cv.equalizeHist(img_dst)

    return img_dst


def rolling_window(orig: np.ndarray, size: int) -> np.ndarray:
    """Generate rolling windows to a 1D array

    Args:
    - `orig:np.ndarray`: Original 1D array
    - `size:int`: Rolling window size

    Returns:
    - 1D Rolling windows
    """
    if orig.size <= size:
        return orig

    shape = orig.shape[:-1] + (orig.shape[-1] - size + 1, size)
    strides = orig.strides + (orig. strides[-1],)

    return np.lib.stride_tricks.as_strided(orig, shape=shape, strides=strides)


def rolling_window_2D(orig: np.ndarray, ref: np.ndarray, delta_x: int=1, delta_y: int=1) -> np.ndarray:
    """Generate rolling windows to a 2D array

    Based on: https://colab.research.google.com/drive/1Zru_-zzbtylgitbwxbi0eDBNhwr8qYl6#scrollTo=R7GoWaxG2RpX

    Args:
    - `orig:np.ndarray`: Original 2D array
    - `ref:np.ndarray`: Reference 2D array to rolling window
    - `delta_x:int`: Horizontal step to find rolling windows
    - `delta_y:int`: Vertical step to find rolling windows

    Returns:
    - 2D Rolling windows
    """
    shape = orig.shape[:-2] + \
            ((orig.shape[-2] - ref.shape[-2]) // delta_y + 1,) + \
            ((orig.shape[-1] - ref.shape[-1]) // delta_x + 1,) + \
            ref.shape  # sausage-like shape with 2D cross-section
    strides = orig.strides[:-2] + \
              (orig.strides[-2] * delta_y,) + \
              (orig.strides[-1] * delta_x,) + \
              orig.strides[-2:]
    return np.lib.stride_tricks.as_strided(orig, shape=shape, strides=strides)


def closest_idx(base: np.ndarray, value: float) -> int:
    """Get last `base` index closest to `value` if a 1D array, otherwise the closest sub-arrays sum to `value`

    Args:
    - `base:np.ndarray`: Search base array
    - `value:float`: Target value

    Returns:
    - Last index if `base` exists, otherwise -1
    """
    if not base.size:
        return -1

    base = np.asarray(base)[::-1]

    if len(base.shape) < 3:
        return len(base) - np.argmin(np.abs(base - value), axis=None) - 1

    return len(base) - np.argmin(np.sum(np.sum(np.abs(base - value), axis=1), axis=1)) - 1


def get_resize_shape(original_shape: tuple) -> tuple:
    """Get target resized shape

    Args:
    - `original_shape:tuple`: Original shape

    Returns:
    - Shape to resize
    """
    from .variables import IMG_LOWEST_DIMENSION

    highest = 'width' if original_shape[0] < original_shape[1] else 'height'

    if highest == 'width':
        return int((IMG_LOWEST_DIMENSION * original_shape[1]) / original_shape[0]), IMG_LOWEST_DIMENSION

    return IMG_LOWEST_DIMENSION, int((IMG_LOWEST_DIMENSION * original_shape[0]) / original_shape[1])


def normalize_map(target: np.ndarray) -> tuple:
    """Normalize a disparity map withing (0 - 254) range, reserving the 255 value to infinity values

    Args:
    - `target:np.ndarray`: Target image to normalize

    Returns:
    - Normalized image and original maximun value
    """
    c_dtype = target.dtype

    target = target.astype(np.float64)

    target[target < 0] = 0

    target_max = np.max(target)

    if target_max == float('inf'):
        target[np.where(target == float('inf')) or np.where(target == float('-inf'))] = np.nan
        target_max = np.nanmax(target)

    target[np.where(target != float('inf')) and np.where(target != float('-inf'))] *= (254 / target_max)
    target[target == np.nan] = 255.

    target = np.round(target, decimals=0)

    return target.astype(c_dtype), target_max


def get_key_pts(img_base: np.ndarray, img_corresp: np.ndarray) -> tuple:
    """Search for sparse key point at the image

    Args:
    - `img_base:np.ndarray`: Base image
    - `img_corresp:np.ndarray`: Correspondent image

    Returns:
    - Both image key points
    """
    from .variables import BEST_MATCH_PERC, FLANN_INDEX_KDTREE, MATCHES_PER_PT

    sift = cv.SIFT_create(contrastThreshold=.007, edgeThreshold=20)

    raw_key_pts_base, descriptors_base = sift.detectAndCompute(img_base, None)
    raw_key_pts_corresp, descriptors_corresp = sift.detectAndCompute(img_corresp, None)

    index_params = {
        'algorithm': FLANN_INDEX_KDTREE,
        'trees': 5
    }
    search_params = {'checks': 50}

    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors_base, descriptors_corresp, k=MATCHES_PER_PT)

    key_pts_base = []
    key_pts_corresp = []
    # ratio test as per Lowe's paper
    for m_L, m_R in matches:
        if m_L.distance < BEST_MATCH_PERC * m_R.distance:
            key_pts_base.append(raw_key_pts_base[m_L.queryIdx].pt)
            key_pts_corresp.append(raw_key_pts_corresp[m_L.trainIdx].pt)

    key_pts_base = np.array(key_pts_base, dtype=np.int32)
    key_pts_corresp = np.array(key_pts_corresp, dtype=np.int32)

    return key_pts_base, key_pts_corresp


def rectify_corresp_imgs(
    img_base: np.ndarray, img_corresp: np.ndarray,
    key_pts_base: np.array, key_pts_corresp: np.array
) -> tuple:
    """Rectify correspondent images considering theirs key points

    Args:
    - `img_base:np.ndarray`: Base image
    - `img_corresp:np.ndarray`: Correspondent image
    - `key_pts_base:np.array`: Base image key points
    - `key_pts_corresp:np.array`: Corresoondent image key points

    Return:
    - Both rectified image
    """
    fund_matrix, mask = cv.findFundamentalMat(key_pts_base, key_pts_corresp, cv.FM_LMEDS)

    # # We select only inlier points
    key_pts_base = key_pts_base[mask.ravel() == 1]
    key_pts_corresp = key_pts_corresp[mask.ravel() == 1]

    _, h1, h2 = cv.stereoRectifyUncalibrated(key_pts_base, key_pts_corresp, fund_matrix, img_base.shape)

    ret_img_base = cv.warpPerspective(img_base, h1, (img_base.shape[1], img_base.shape[0]))
    ret_img_corresp = cv.warpPerspective(img_corresp, h2, (img_corresp.shape[1], img_corresp.shape[0]))

    return ret_img_base, ret_img_corresp


def _parse_intrinsic(raw_line: str) -> np.array:
    """Parse a intrinsic matrix 1-line string ("[f 0 cx; 0 f cy; 0 0 1]") to array

    Args:
    - `raw_line:str`: intrinsic matrix 1-line string

    Returns:
    - Parsed array
    """
    return np.array([
        list(map(float, row.strip().split()))
        for row in raw_line.replace('[', '').replace(']', '').split(';')
    ], dtype='float32')
