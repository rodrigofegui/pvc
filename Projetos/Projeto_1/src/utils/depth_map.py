import cv2 as cv
import matplotlib.pyplot as plotter
import numpy as np


def get_depth_map(calib, disp_map):
    depth_map = np.full_like(disp_map, fill_value=calib['intrinsic_0'][0][0] * calib['baseline'], dtype='float64')

    indexes = np.where(disp_map != 0)

    depth_map[indexes] = depth_map[indexes] / disp_map[indexes]
    depth_map[disp_map == 0] = 0

    return depth_map


def calc_depth_map(calib, disp_map, method_name='', ratio=1, file_name=None):
    depth_map = get_depth_map(calib, disp_map)
    norm_depth_map = cv.normalize(depth_map, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
    eq_depth_map = cv.equalizeHist(norm_depth_map)

    min_dist = np.round(np.min(depth_map) * .0001 * ratio, decimals=2)
    max_dist = np.round(np.max(depth_map) * .0001 * ratio, decimals=2)

    plotter.colorbar(plotter.imshow(eq_depth_map, cmap='gist_heat'))
    plotter.suptitle(f'Mapa de profundidade: {method_name}', fontsize=14, y=.95)
    plotter.title(f'Norm. e eq.: {min_dist} - {max_dist} m', fontsize=8, y=1.08, x=.55)

    if file_name:
        plotter.savefig(file_name)

    plotter.clf()
