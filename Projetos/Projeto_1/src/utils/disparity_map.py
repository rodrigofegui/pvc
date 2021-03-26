import cv2 as cv
import numpy as np

from .utils import closest_idx, rolling_window, rolling_window_2D
from .variables import ACCEPT_DIFF, DISP_BLOCK_SEARCH, MIN_DISP_FILTER_SZ


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
    block_sz = int(block_sz / 2)

    for y in range(img_left.shape[0]):
        print(f'linha: {y}', end='\r')
        for x_l in range(img_left.shape[1]):
            y_min, y_max = max(0, y - block_sz), min(y + block_sz + 1, img_left.shape[0])
            x_min, x_max = max(0, x_l - block_sz), min(x_l + block_sz + 1, img_left.shape[1])

            x_match = x_l

            ref = img_left[y_min:y_max, x_min:x_max]

            try:
                search = rolling_window_2D(img_right[y:y + ref.shape[0], :x_max], ref)

                if search.size:
                    search = search[0] - ref
                    x_match = int(closest_idx(search, 0) / ref.size)
            except Exception as e:
                print('error', e)
            finally:
                disp_map[y, x_l] = x_l - x_match

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
        print(f'linha: {unpadded_y}', end='\r')

        for unpadded_x, x_l in enumerate(range(padding, img_left.shape[1] - padding - 1)):
            x_min = max(padding, x_l - lookup_size)
            x_max = min(x_l + block_sz if x_l > block_sz else x_l + 1, img_left.shape[1])

            ref = img_left[y, x_l:x_max]
            search = img_right[y, x_min:x_max]

            # print(f'ref: {ref.shape}\n{ref}')

            # print(f'search: {search.shape}\n{search}')
            search = rolling_window(search, ref.size)
            x_correlation = np.dot(search, ref) / (np.linalg.norm(search, axis=1) * np.linalg.norm(ref))
            x_correlation = np.round(x_correlation, decimals=4)

            # print(f'x_correlation: {x_correlation.shape}\n{x_correlation}')
            x_match = closest_idx(x_correlation, 1)

            if x_match == -1:
                disp_map[unpadded_y, x_l] = -1
                continue

            x_match += x_min

            # print(f'\tx: {x_l} | x_min: {x_min} | x_r: {x_match}')
            disp_map[unpadded_y, unpadded_x] = x_l - x_match if x_match <= x_l else -1

        #     print()

        #     if unpadded_x >= block_sz + 1:
        #         break

        # break

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
        print(f'linha: {y}', end='\r')
        for cnt, x_l in enumerate(range(img_left.shape[1] - 1, -1, -1)):
            x_min = max(0, x_l - lookup_size)

            ref = img_left[y, x_l:min(x_l + block_sz if x_l > block_sz else x_l + 1, img_left.shape[1])]

            search = rolling_window(img_right[y, x_min:x_l + block_sz - 1], ref.size).astype(np.int32) - ref

            x_match = int(closest_idx(search, 0) / ref.size)

            if x_match == -1:
                disp_map[y, x_l] = -1
                continue

            x_match += x_min

            disp_map[y, x_l] = x_l - x_match if x_match <= x_l else -1

    return disp_map
