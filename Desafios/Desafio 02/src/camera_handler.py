from glob import glob
from random import randint

import cv2 as cv
import numpy as np


def get_real_and_img_pts(
    pattern_size: tuple,
    img_name_pattern: str,
    real_square_size: float = 1.0,
    show_progress: bool = False
) -> tuple:
    """Get 3D points in real world space, 2D points in image place and the shape of the used images.
    Adaptation from: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

    Args:
    - `pattern_size:tuple`: Chess pattern grid to be considerer as `(width, height)`
    - `img_name_pattern:str`: Image name pattern that contains a chessboard
    - `real_square_size:float`: Real measure from the squares into the chessboard
    - `show_progress:bool`: Control the image exhibition as they been process

    Returns:
    - (`real_pts`, `img_pts`, `img_gray.shape`, `img_names`)
    """
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, .0001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    real_pt = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    real_pt[:, :2] = np.mgrid[:pattern_size[1], :pattern_size[0]].T.reshape(-1, 2)
    # real square measure
    real_pt *= real_square_size

    # 3D points and 2D points
    real_pts, img_pts = [], []
    img_gray = np.zeros((1, 1))

    img_names = glob(img_name_pattern)
    for img_name in img_names:
        img = cv.imread(img_name)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        status_ok, corners = cv.findChessboardCorners(
            img_gray, pattern_size,
            flags=cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_ADAPTIVE_THRESH
        )

        if not status_ok:
            continue

        real_pts.append(real_pt)
        img_pts.append(corners)

        if show_progress:
            corners2 = cv.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
            cv.drawChessboardCorners(img, pattern_size, corners2, True)
            cv.imshow('img', img)
            cv.waitKey(500)

    cv.destroyAllWindows()

    return real_pts, img_pts, img_gray.shape, img_names


def get_camera_matrices(real_pts: list, img_pts: list, img_shape: tuple) -> tuple:
    """Get general camera matrices: intrinsic params and distortion coefficients; and
    specific camera matrices: rotation, translation and projection.

    Source: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

    Args:
    - `real_pts:list`: 3D points in real world space
    - `img_pts:list`: 2D points in image place
    - `img_shape:tuple`: Used images' shape

    Returns:
    - Matrices in dict
    """
    _, intrinsics, distortion, rot_vecs, trans_vecs = cv.calibrateCamera(
        real_pts, img_pts, img_shape[::-1], None, None
    )

    pos = randint(0, len(rot_vecs) - 1)
    rotation, _ = cv.Rodrigues(rot_vecs[pos])
    translation = trans_vecs[pos]

    extrinsic = cv.hconcat((rotation, translation))
    projection = np.matmul(intrinsics, extrinsic)

    return {
        'intrinsics': intrinsics,
        'distortion': distortion,
        'specific': {
            'sorted': pos,
            'rotation': rotation,
            'translation': translation,
            'extrinsic': extrinsic,
            'projection': projection
        }
    }
