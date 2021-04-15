import cv2 as cv
import numpy as np
import matplotlib.pyplot as plotter

from .utils import closest_idx, rolling_window, rolling_window_2D
from .variables import MAX_DISP_LOOKUP, MIN_DISP_FILTER_SZ


def get_diff_percent(target_img: np.ndarray, base_img: np.ndarray, threshold: float) -> tuple:
    """Calculate the diff ratio between two images, usually disparity maps, based on threshold value.

    Args:
    - `target_img:np.ndarray`: Target image to calculate the error
    - `base_img:np.ndarray`: Base image
    - `threshold:float`: Diff threshold

    Returns:
    - Diff ratio
    - Raw diff image
    """
    diff = target_img - base_img
    errors = np.abs(diff) > threshold

    return np.round(np.count_nonzero(errors) / base_img.size, decimals=4), diff


def save_disp_map(disp_map: np.ndarray, title_detail: str='-', file_name: str=None) -> None:
    plotter.colorbar(plotter.imshow(disp_map, cmap='gist_heat'))
    plotter.suptitle(f'Mapa de disparidade: {title_detail}', fontsize=14, y=.95)

    if file_name:
        plotter.savefig(file_name)
    else:
        plotter.show()

    plotter.clf()


def basic_disp_map(
    img_left:np.ndarray,
    img_right:np.ndarray,
    block_sz:int = MIN_DISP_FILTER_SZ,
    lookup_size:int = MAX_DISP_LOOKUP
) -> np.ndarray:
    """Disparity map using OpenCV default method, after empirical tests

    Based on: https://gist.github.com/andijakl/ffe6e5e16742455291ef2a4edbe63cb7

    Args:
    - `img_left:np.ndarray`: Left image
    - `img_right:np.ndarray`: Right image
    - `block_sz:int`: Filter size, default value `MIN_DISP_FILTER_SZ`
    - `lookup_size:int`: Superior limit to lookup, default value `MAX_DISP_LOOKUP`

    Returns:
    - Disparity map
    """
    # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
    # Normally, a value within the 5-15 range is good enough
    uniquenessRatio = 10
    # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
    # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    speckleWindowSize = 200
    # Maximum disparity variation within each connected component.
    # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
    # Normally, 1 or 2 is good enough.
    speckleRange = 1
    disp12MaxDiff = 0
    # Lambda is a parameter defining the amount of regularization during filtering
    wls_lambda=.5
    # SigmaColor is a parameter defining how sensitive the filtering process is to source
    # image edges. Large values can lead to disparity leakage through low-contrast edges.
    wls_sigma_color=80

    stereo_L = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=lookup_size,
        blockSize=block_sz,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_sz * block_sz,
        P2=32 * 1 * block_sz * block_sz,
    )
    stereo_R = cv.ximgproc.createRightMatcher(stereo_L)
    wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo_L)
    wls_filter.setLambda(wls_lambda)
    wls_filter.setSigmaColor(wls_sigma_color)

    disp_map_L = stereo_L.compute(img_left, img_right)
    disp_map_R = stereo_R.compute(img_right, img_left)
    disp_map = wls_filter.filter(disp_map_L, img_left, disparity_map_right=disp_map_R, right_view=img_right)

    return disp_map


def windowing_disp_map(
    img_left:np.ndarray,
    img_right:np.ndarray,
    block_sz:int = MIN_DISP_FILTER_SZ,
    lookup_size:int = MAX_DISP_LOOKUP
) -> np.ndarray:
    """Disparity map using windowing comparison method

    Args:
    - `img_left:np.ndarray`: Left image
    - `img_right:np.ndarray`: Right image
    - `block_sz:int`: Filter size, default value `MIN_DISP_FILTER_SZ`
    - `lookup_size:int`: Superior limit to lookup, default value `MAX_DISP_LOOKUP`

    Returns:
    - Disparity map
    """
    disp_map = np.zeros_like(img_left)
    padding = int(block_sz / 2)

    accept_diff_base = 12

    img_left = np.pad(
        img_left, [(padding, padding), (padding, padding)], mode='constant', constant_values=0
    ).astype(np.int32)
    img_right = np.pad(
        img_right, [(padding, padding), (padding, padding)], mode='constant', constant_values=0
    ).astype(np.int32)

    for unpadded_y, y in enumerate(range(padding, img_left.shape[0] - padding)):
        # print(f'linha: {unpadded_y}', end='\r')

        for unpadded_x, x_l in enumerate(range(padding, img_left.shape[1] - padding)):
            y_min, y_max = max(0, y - padding), min(y + padding + 1, img_left.shape[0])
            x_l_min, x_l_max = max(0, x_l - padding), min(x_l + padding + 1, img_left.shape[1])
            x_r_min = max(0, x_l - lookup_size)

            ref = img_left[y_min:y_max, x_l_min:x_l_max]

            search = img_right[y_min:y_max, x_r_min:x_l_max]
            search = (rolling_window_2D(search, ref).astype(np.int32))[0] - ref

            x_match = int(closest_idx(search, 0))

            if x_match != -1:
                try:
                    search = np.sum(search[x_match])
                    x_match = x_match + x_r_min + padding if search <= (accept_diff_base * ref.size) else -1
                except:
                    input(f'match: {x_match} | search:\n{search}')
                    x_match = -1

            disp_map[unpadded_y, unpadded_x] = x_l - x_match if x_match <= x_l and x_match != 1 else -1

    return _minimize_invalid_pxl(disp_map)


def x_correlation_disp_map(
    img_left:np.ndarray,
    img_right:np.ndarray,
    block_sz:int = MIN_DISP_FILTER_SZ,
    lookup_size:int = MAX_DISP_LOOKUP
) -> np.ndarray:
    """Disparity map using cross correlation method

    Args:
    - `img_left:np.ndarray`: Left image
    - `img_right:np.ndarray`: Right image
    - `block_sz:int`: Filter size, default value `MIN_DISP_FILTER_SZ`
    - `lookup_size:int`: Superior limit to lookup, default value `MAX_DISP_LOOKUP`

    Returns:
    - Disparity map
    """
    disp_map = np.zeros_like(img_left)

    padding = int(block_sz / 2)
    x_corr_accept_diff = .15

    img_left = np.pad(
        img_left, [(padding, padding), (padding, padding)], mode='constant', constant_values=0
    ).astype(np.int32)
    img_right = np.pad(
        img_right, [(padding, padding), (padding, padding)], mode='constant', constant_values=0
    ).astype(np.int32)

    for unpadded_y, y in enumerate(range(padding, img_left.shape[0] - padding)):
        # print(f'linha: {unpadded_y}', end='\r')

        for unpadded_x, x_l in enumerate(range(padding, img_left.shape[1] - padding - 1)):
            x_min = max(0, x_l - lookup_size)
            x_max = min(x_l + block_sz, img_left.shape[1])

            ref = img_left[y, x_l:x_max]
            search = img_right[y, x_min:x_max]

            # print(f'ref:\n{ref}')
            search = rolling_window(search, ref.size)
            # print(f'search:\n{search}')
            # print(f'|search|:\n{np.linalg.norm(search, axis=1)}')
            x_correlation = np.dot(search, ref) / (np.linalg.norm(search, axis=1) * np.linalg.norm(ref))
            x_correlation = np.round(x_correlation, decimals=4)
            # print(f'x_correlation:\n{x_correlation}')

            x_match = closest_idx(x_correlation, 1)

            if x_match != -1:
                # print(f'\tx_correlation[{x_match}]: {x_correlation[x_match]} | {np.sum(search[x_match] - ref)}')
                x_match = x_match + x_min + padding if 1 - x_correlation[x_match] <= x_corr_accept_diff else -1

            # input()

            disp_map[unpadded_y, unpadded_x] = x_l - x_match if x_match <= x_l and x_match != 1 else -1

    return _minimize_invalid_pxl(disp_map)


def linear_search_disp_map(
    img_left:np.ndarray,
    img_right:np.ndarray,
    block_sz:int = MIN_DISP_FILTER_SZ,
    lookup_size:int = MAX_DISP_LOOKUP
) -> np.ndarray:
    """Disparity map using single linear search method

    Args:
    - `img_left:np.ndarray`: Left image
    - `img_right:np.ndarray`: Right image
    - `block_sz:int`: Filter size, default value `MIN_DISP_FILTER_SZ`
    - `lookup_size:int`: Superior limit to lookup, default value `MAX_DISP_LOOKUP`

    Returns:
    - Disparity map
    """
    disp_map = np.zeros_like(img_left)

    accept_diff_base = 12

    for y in range(img_left.shape[0]):
        # print(f'linha: {y}', end='\r')
        for x_l in range(img_left.shape[1]):
            x_min = max(0, x_l - lookup_size)
            x_l_max = min(x_l + block_sz if x_l > block_sz else x_l + 1, img_left.shape[1])
            x_r_max = min(x_l + block_sz - 1, img_left.shape[1])

            ref = img_left[y, x_l:x_l_max]

            search_box = rolling_window(img_right[y, x_min:x_r_max], ref.size).astype(np.int32) - ref

            if search_box.size <= 1:
                x_match = x_l - 1
            else:
                x_match = int(closest_idx(search_box, 0) / ref.size)
                search = np.sum(search_box[x_match])

                x_match = x_match + x_min if search <= (accept_diff_base * ref.size) else -1

            disp_map[y, x_l] = max(0, x_l - x_match) if x_match != 1 else -1

    return _minimize_invalid_pxl(disp_map)


def slow_adaptative_windowing_disp_map(
    img_left:np.ndarray,
    img_right:np.ndarray,
    block_sz:int = MIN_DISP_FILTER_SZ,
    lookup_size:int = MAX_DISP_LOOKUP
) -> np.ndarray:
    """Disparity map using windowing method with raw python, so it can takes a while

    Args:
    - `img_left:np.ndarray`: Left image
    - `img_right:np.ndarray`: Right image
    - `block_sz:int`: Filter size, default value `MIN_DISP_FILTER_SZ`
    - `lookup_size:int`: Superior limit to lookup, default value `MAX_DISP_LOOKUP`

    Returns:
    - Disparity map
    """
    padding = int(block_sz / 2)

    disp_map = np.zeros_like(img_left)
    matched = np.zeros_like(img_left)

    for y in range(img_left.shape[0]):
        # print(f'linha: {y}')
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
                except:
                    pass

                if cmp_sum < sum_match:
                    # print(f'\t{x_l} -> {x_r}'):
                    sum_match = cmp_sum
                    x_match = x_r

            # input('waiting...')

            # print(f'\t{x_l} -> {x_match}')
            disp_map[y, x_l] = (x_l - x_match) or 255
            matched[y, x_match] = 1

    return disp_map


def _minimize_invalid_pxl(orig: np.ndarray) -> np.ndarray:
    """Replace invalid pixels (marked as -1) by the 9-neighbors mean

    Args:
    - `orig:np.ndarray`: Original image with invalid pixels

    Returns:
    - Handled image
    """
    orig[orig == -0] = 0

    if np.min(orig) != -1:
        return orig

    padding = 1

    orig = np.pad(
        orig, [(padding, padding), (padding, padding)], mode='constant', constant_values=256
    ).astype(np.float32)
    orig[orig == 256] = np.nan

    invalid_pxl_replacements = np.round(np.array(
        np.nanmean(
            np.array([
                orig[:-2, :-2],
                orig[1:-1, :-2],
                orig[2:, :-2],

                orig[:-2, 1:-1],
                orig[1:-1, 1:-1],
                orig[2:, 1:-1],

                orig[:-2, 2:],
                orig[1:-1, 2:],
                orig[2:, 2:],
            ]),
            axis=0,
            keepdims=True
        )[0]
    ), decimals=0)

    invalid_pxl_replacements[invalid_pxl_replacements == -1] = 255

    orig = orig[1:-1, 1:-1]

    idxs = np.where(orig == -1)
    orig[idxs] = invalid_pxl_replacements[idxs]

    return orig
