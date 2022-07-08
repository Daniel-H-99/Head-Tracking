#!/usr/bin/python
from rsa import verify
import numpy as np
import cv2
from skimage import io
import helpers
from matplotlib import pyplot as plt
import face_alignment
import argparse
import os

######################################################
## Landmark detection with one image.               ##
## The result is saved as a PDF file.               ##
######################################################

parser = argparse.ArgumentParser(description='extract keypoints from image')

parser.add_argument('--source_path', default='./datasample/Amir.jpg', help='input sequence path')
parser.add_argument('--out_dir', default='./results/', help='FLAME meshes output path')

args = parser.parse_args()

######################## read image ################################
# For one image
path = args.source_path
output_name = path.split('/')[-1].split('.jpg')[0].split('.png')[0]
output_name = os.path.join(args.out_dir, output_name)

# NAME = 'Amir'
# path = f"./dataSample/{NAME}.jpg"
im = io.imread(path)

# im = io.imread('/root/mnt/share/Voxceleb/train/id10841#5h5NmY820_M#001845#001980.mp4/0000000.png')
H, W = im.shape[:2]
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
# bb = (0, 0, H - 1, W - 1)
# coords = fa.get_landmarks(im)[0].astype(int)
# frame_landmarks = np.stack([frame_landmarks[:, 0].clip(0, H - 1), frame_landmarks[:, 1].clip(0, W - 1)], axis=1)
# (vertices, mesh_plotting, Ind, rotation_angle) = helpers.landmarks_3d_fitting(frame_landmarks,height,width)

(bb, coords) = helpers.get_landmarks_fa(im)
# print(f'bb: {bb}')
# print(f'coords: {coords}')
# (vertices, mesh_plotting, Ind, rotation_angle) = helpers.landmarks_3d_fitting(coords, H, W)

vertices, a, Ind = helpers.normalize_mesh(coords, H, W)


# vertices_rhs = np.load('../flame-fitting/output_landmark/fit_landmarks.npy') * 1000

# print(f'coords: {vertices[-51:]}')/
# print(f'coords_rhs: {vertices_rhs.astype(int)})')
# print(f'coords diff: {vertices_rhs - vertices[-51:]}')

# vertices[-51:] = vertices_rhs
coords_3d = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1) @ a
# print(f'final coords: {coords}')
coords = coords_3d[:, :2].astype(int) + np.array([[W // 2, H // 2]])

#np.savetxt('landmark_result_Yu.txt', coords)

# coords = mesh_plotting[Ind, :2].astype(int) + np.array([[W // 2, H // 2]])
################### output landmark coordinates ####################
for i in [0,16,36,43]:#np.arange(68):
    print("The " +  str(i) + "th landmark:")
    print(coords[i])
  

######################## visualization #############################
highlights = [16,36,43]
outImg = helpers.visualize_facial_landmarks(im, bb, coords, 1, highlights)
outImg_noBackground = helpers.visualize_facial_landmarks(im, bb, coords, 0 ,highlights)

plt.subplot(121), plt.imshow(outImg)
plt.subplot(122), plt.imshow(outImg_noBackground)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

fig = plt.figure()
outImg1 = helpers.visualize_facial_landmarks(im, bb, coords, 1,[])
plt.imshow(outImg1)

plt.show()

np.save(f'{output_name}_vertices.npy', vertices)
np.save(f'{output_name}_pose.npy', coords_3d)
fig.savefig(f'{output_name}_plot.png')
