import cv2 as cv
import numpy as np

from .utils import closest_idx, rolling_window, rolling_window_2D
from .variables import DISP_BLOCK_SEARCH, MIN_DISP_FILTER_SZ


def basic_disp_map(
    img_left:np.ndarray,
    img_right:np.ndarray,
    block_sz:int = MIN_DISP_FILTER_SZ,
    lookup_size:int = DISP_BLOCK_SEARCH
) -> np.ndarray:
    """Disparity map using OpenCV default method

    Args:
    - `img_left:np.ndarray`: Left image
    - `img_right:np.ndarray`: Right image
    - `block_sz:int`: Filter size, default value `MIN_DISP_FILTER_SZ`
    - `lookup_size:int`: Superior limit to lookup, default value `DISP_BLOCK_SEARCH`

    Returns:
    - Disparity map
    """
    return cv.StereoBM_create(
        numDisparities=lookup_size,
        blockSize=block_sz
    ).compute(img_left, img_right)


def windowing_disp_map(
    img_left:np.ndarray,
    img_right:np.ndarray,
    block_sz:int = MIN_DISP_FILTER_SZ,
    lookup_size:int = DISP_BLOCK_SEARCH
) -> np.ndarray:
    """Disparity map using windowing comparison method

    Args:
    - `img_left:np.ndarray`: Left image
    - `img_right:np.ndarray`: Right image
    - `block_sz:int`: Filter size, default value `MIN_DISP_FILTER_SZ`
    - `lookup_size:int`: Superior limit to lookup, default value `DISP_BLOCK_SEARCH`

    Returns:
    - Disparity map
    """
    disp_map = np.zeros_like(img_left)
    padding = int(block_sz / 2)

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

            if x_match == -1:
                disp_map[unpadded_y, unpadded_x] = -1
                continue

            x_match += padding

            disp_map[unpadded_y, unpadded_x] = x_l - x_match if x_match <= x_l else -1

    return disp_map


def x_correlation_disp_map(
    img_left:np.ndarray,
    img_right:np.ndarray,
    block_sz:int = MIN_DISP_FILTER_SZ,
    lookup_size:int = DISP_BLOCK_SEARCH
) -> np.ndarray:
    """Disparity map using cross correlation method

    Args:
    - `img_left:np.ndarray`: Left image
    - `img_right:np.ndarray`: Right image
    - `block_sz:int`: Filter size, default value `MIN_DISP_FILTER_SZ`
    - `lookup_size:int`: Superior limit to lookup, default value `DISP_BLOCK_SEARCH`

    Returns:
    - Disparity map
    """
    disp_map = np.zeros_like(img_left)

    padding = int(block_sz / 2)

    img_left = np.pad(
        img_left, [(padding, padding), (padding, padding)], mode='constant', constant_values=0
    ).astype(np.int32)
    img_right = np.pad(
        img_right, [(padding, padding), (padding, padding)], mode='constant', constant_values=0
    ).astype(np.int32)

    for unpadded_y, y in enumerate(range(padding, img_left.shape[0] - padding)):
        # print(f'linha: {unpadded_y}', end='\r')

        for unpadded_x, x_l in enumerate(range(padding, img_left.shape[1] - padding - 1)):
            x_min = max(padding, x_l - lookup_size)
            x_max = min(x_l + block_sz if x_l > block_sz else x_l + 1, img_left.shape[1])

            ref = img_left[y, x_l:x_max]
            search = img_right[y, x_min:x_max]

            search = rolling_window(search, ref.size)
            x_correlation = np.dot(search, ref) / (np.linalg.norm(search, axis=1) * np.linalg.norm(ref))
            x_correlation = np.round(x_correlation, decimals=4)

            x_match = closest_idx(x_correlation, 1)

            if x_match == -1:
                disp_map[unpadded_y, unpadded_x] = -1
                continue

            x_match += x_min

            disp_map[unpadded_y, unpadded_x] = x_l - x_match if x_match <= x_l else -1

    return disp_map


def linear_search_disp_map(
    img_left:np.ndarray,
    img_right:np.ndarray,
    block_sz:int = MIN_DISP_FILTER_SZ,
    lookup_size:int = DISP_BLOCK_SEARCH
) -> np.ndarray:
    """Disparity map using single linear search method

    Args:
    - `img_left:np.ndarray`: Left image
    - `img_right:np.ndarray`: Right image
    - `block_sz:int`: Filter size, default value `MIN_DISP_FILTER_SZ`
    - `lookup_size:int`: Superior limit to lookup, default value `DISP_BLOCK_SEARCH`

    Returns:
    - Disparity map
    """
    disp_map = np.zeros_like(img_left)

    for y in range(img_left.shape[0]):
        # print(f'linha: {y}', end='\r')
        for cnt, x_l in enumerate(range(img_left.shape[1] - 1, -1, -1)):
            x_min = max(0, x_l - lookup_size)
            x_l_max = min(x_l + block_sz if x_l > block_sz else x_l + 1, img_left.shape[1])
            x_r_max = min(x_l + block_sz - 1, img_left.shape[1])

            ref = img_left[y, x_l:x_l_max]

            search = rolling_window(img_right[y, x_min:x_r_max], ref.size).astype(np.int32) - ref

            x_match = int(closest_idx(search, 0) / ref.size)

            if x_match == -1:
                disp_map[y, x_l] = -1
                continue

            x_match += x_min

            disp_map[y, x_l] = x_l - x_match if x_match <= x_l else -1

    return disp_map


def judged_windowing_disp_map(
    img_left:np.ndarray,
    img_right:np.ndarray,
    block_sz:int = MIN_DISP_FILTER_SZ,
    lookup_size:int = DISP_BLOCK_SEARCH
) -> np.ndarray:
    disp_map = np.zeros_like(img_left).astype(np.int32)
    padding = 1

    accept_diff = 10
    accept_diff *= (((padding * 2) + 1) ** 2)

    img_left = np.pad(
        img_left, [(padding, padding), (padding, padding)], mode='constant', constant_values=0
    ).astype(np.int32)
    img_right = np.pad(
        img_right, [(padding, padding), (padding, padding)], mode='constant', constant_values=0
    ).astype(np.int32)
    print(f'lookup_size: {lookup_size}')

    for unpadded_y, y in enumerate(range(padding, img_left.shape[0] - padding)):
        print(f'linha: {unpadded_y}', end='\r')

        y_min, y_max = max(0, y - padding), min(y + padding + 1, img_left.shape[0])

        for unpadded_x, x_l in enumerate(range(padding, img_left.shape[1] - padding)):
            x_l_min, x_l_max = max(0, x_l - padding), min(x_l + padding + 1, img_left.shape[1])
            x_r_min = max(0, x_l_min - lookup_size)

            ref = img_left[y_min:y_max, x_l_min:x_l_max]
            search_box = img_right[y_min:y_max, x_r_min:x_l_max]
            search_box = rolling_window_2D(search_box, ref)[0]

            if search_box.shape[0] == 1:
                x_match = -1
            else:
                search = np.sum(np.sum(np.abs(search_box - ref), axis=1), axis=1)
                x_match = closest_idx(search, 0)
                search = search[x_match]

                x_match = x_match + x_r_min if search <= accept_diff else -1

            disp_map[unpadded_y, unpadded_x] = x_l - x_match if x_match <= x_l and x_match != 1 else -1

    return _minimize_invalid_pxl(disp_map)


def _minimize_invalid_pxl(disp_map):
    disp_map[disp_map == -0] = 0

    if np.min(disp_map) != -1:
        return disp_map

    padding = 1

    disp_map = np.pad(
        disp_map, [(padding, padding), (padding, padding)], mode='constant', constant_values=256
    ).astype(np.float32)
    disp_map[disp_map == 256] = np.nan

    invalid_pxl_replacements = np.round(np.array(
        np.nanmean(
            np.array([
                disp_map[:-2, :-2],
                disp_map[1:-1, :-2],
                disp_map[2:, :-2],

                disp_map[:-2, 1:-1],
                disp_map[1:-1, 1:-1],
                disp_map[2:, 1:-1],

                disp_map[:-2, 2:],
                disp_map[1:-1, 2:],
                disp_map[2:, 2:],
            ]),
            axis=0,
            keepdims=True
        )[0]
    ), decimals=0)

    invalid_pxl_replacements[invalid_pxl_replacements == -1] = 255

    disp_map = disp_map[1:-1, 1:-1]

    idxs = np.where(disp_map == -1)
    disp_map[idxs] = invalid_pxl_replacements[idxs]

    return disp_map
