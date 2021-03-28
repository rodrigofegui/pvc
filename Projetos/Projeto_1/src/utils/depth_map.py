import cv2 as cv
import numpy as np

def get_depth_map(calib, disp_map):
    depth_map = np.full_like(disp_map, fill_value=calib['intrinsic_0'][0][0] * calib['baseline'])

    return depth_map / disp_map
