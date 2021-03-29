import cv2 as cv
import matplotlib.pyplot as plotter
import numpy as np
from re import findall
from utils.utils import normalize_map


def get_depth_map(calib, disp_map):
    depth_map = np.full_like(disp_map, fill_value=calib['intrinsic_0'][0][0] * calib['baseline']).astype('float32')

    indexes = np.where(disp_map != 0)

    depth_map[indexes] = depth_map[indexes] / disp_map[indexes]
    depth_map[disp_map == 0] = float('inf')

    return depth_map


def calc_depth_map(calib, file_names, result_dir):
    for cnt, disp_map_name in enumerate(file_names):
        print(f'calculating depth map to: {disp_map_name}')
        target_name = disp_map_name.split("/")[-1][:-4]

        # depth_map = cv.imread(disp_map_name, cv.IMREAD_UNCHANGED)
        disp_map = cv.imread(disp_map_name, cv.IMREAD_UNCHANGED)
        depth_map = get_depth_map(calib, disp_map)
        # depth_map, depth_map_max = normalize_map(depth_map)
        print(f'depth_map: {depth_map.shape} | min: {np.min(depth_map)} | max: {np.max(depth_map)}')

        color_map = plotter.get_cmap('jet')(depth_map)
        # print(f'color_map: {color_map.shape}\n{color_map}')

        plotter.imshow(color_map)
        plotter.colorbar()
        plotter.clim(np.min(depth_map), np.max(depth_map))
        plotter.title('Mapa de profundidade')
        # plotter.xlabel()
        plotter.savefig(f'{result_dir}/{target_name}_color.png')
        plotter.clf()

        if cnt >= 10: break
