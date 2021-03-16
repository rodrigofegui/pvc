from src.camera_handler import get_camera_matrices, get_real_and_img_pts

CHESS_GRID_SHAPE = (7, 7)
REAL_CHESS_SQ_SIZE = 2.1
IMG_CHESS_SRC_PATTERN = './images/*.jpg'
SHOW_CHESS_ANALYSES = False

real_pts, img_pts, img_shape, img_names = get_real_and_img_pts(
    CHESS_GRID_SHAPE, IMG_CHESS_SRC_PATTERN, REAL_CHESS_SQ_SIZE, SHOW_CHESS_ANALYSES
)

matrices = get_camera_matrices(real_pts, img_pts, img_shape)

print('Intrinsic Matrix (K):')
print(matrices['intrinsics'], end='\n\n')

print('Distortion Coefficients:')
print(matrices['distortion'], end='\n\n')

sorted_pos = matrices['specific']['sorted']
print(f'Data based on the image: "{img_names[sorted_pos]}"')
print('Rotation Matrix (R):')
print(matrices['specific']['rotation'], end='\n\n')

print('Translation Matrix (t):')
print(matrices['specific']['translation'], end='\n\n')

print('Extrinsic Matrix ([R | t]):')
print(matrices['specific']['extrinsic'], end='\n\n')

print('Projection Matrix (P):')
print(matrices['specific']['projection'], end='\n\n')
