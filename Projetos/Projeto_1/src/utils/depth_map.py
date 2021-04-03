import cv2 as cv
import matplotlib.pyplot as plotter
import numpy as np


def get_depth_map(calib: dict, disp_map: np.ndarray) -> np.ndarray:
    """Get depth map to a dispiraty map considering calibration info

    Args:
    - `calib:dict`: Challenge calibration info
    - `disp_map:np.ndarray`: Original disparity map

    Returns:
    - Calculated depth map
    """
    depth_map = np.full_like(disp_map, fill_value=calib['intrinsic_0'][0][0] * calib['baseline'], dtype='float64')

    indexes = np.where(disp_map != 0)

    depth_map[indexes] = depth_map[indexes] / disp_map[indexes]
    depth_map[disp_map == 0] = 0

    return depth_map


def draw_depth_map(
    calib: dict, disp_map: np.ndarray,
    title_detail: str='-', ratio_sz: float=1,
    file_name: str=None
) -> None:
    """Draw depth map considering a disparity map either saving it or showing it

    Args:
    - `calib:dict`: Challenge calibration info
    - `disp_map:np.ndarray`: Original disparity map
    - `title_detail:str`: Dispiraty map origem
    - `ratio_sz:float`: Resize ratio to be compensated
    - `file_name:str`: File name to save the depth map
    """
    depth_map = get_depth_map(calib, disp_map)
    norm_depth_map = cv.normalize(depth_map, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
    eq_depth_map = cv.equalizeHist(norm_depth_map)

    min_dist = np.round(np.min(depth_map) * .0001 * ratio_sz, decimals=2)
    max_dist = np.round(np.max(depth_map) * .0001 * ratio_sz, decimals=2)

    plotter.colorbar(plotter.imshow(eq_depth_map, cmap='gist_heat'))
    plotter.suptitle(f'Mapa de profundidade: {title_detail}', fontsize=14, y=.95)
    plotter.title(f'Norm. e eq.: {min_dist} - {max_dist} m', fontsize=8, y=1.08, x=.55)

    if file_name:
        plotter.savefig(file_name)
    else:
        plotter.show()

    plotter.clf()
