import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec
import time
import os    
import argparse
from skimage import io
import helpers

parser = argparse.ArgumentParser(description='extract keypoints from image')

parser.add_argument('--source_path', default='../voca/animation_output_vox_a_f/landmarks.npy', help='input sequence path')
parser.add_argument('--out_path', default='./results/landmark_video.mp4', help='FLAME meshes output path')
parser.add_argument('--background_path', default=None, help='FLAME meshes output path')
parser.add_argument('--H', default=256, help='FLAME meshes output path')
parser.add_argument('--W', default=256, help='FLAME meshes output path')
parser.add_argument('--fps', default=25, help='FLAME meshes output path')
# parser.add_argument('--W', default=256, help='FLAME meshes output path')

args = parser.parse_args()

######################## read image ################################
# For one image
path = args.source_path
output_name = args.out_path
fps = args.fps

if args.background_path is not None:
    bg = io.imread(args.background_path)
    H, W = bg.shape[:2]
    canvas = bg
    use_bg = True
else:
    height = args.H 
    width = args.W
    canvas = np.zeros((height, width, 3)).astype(np.uint8)
    use_bg = False
landmarks = np.load(path) * 1000 # B x N x 3



fourcc = cv2.VideoWriter_fourcc(*'mp4v') # create VideoWriter object
# width, height = 256, 256
out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
    
for i, lm in enumerate(landmarks):
    ######################## visualization #############################
    # plt.subplot(11)
    # ax2 = plt.subplot(111)
    # ax2.imshow(im2),ax2.set_title('Landmark Extraction on Raw Data', fontsize=16)
    print(f'lm shape: {lm.shape}')
    frame = helpers.visualize_facial_landmarks(canvas, (0, 0, height - 1, width - 1), lm[:, :2].astype(int) + np.array([[width // 2, height // 2]]))
    out.write(frame)
out.release()
