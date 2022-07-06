#!/usr/bin/python
import numpy as np
import cv2
from skimage import io
import helpers
from matplotlib import pyplot as plt
import face_alignment

######################################################
## Landmark detection with one image.               ##
## The result is saved as a PDF file.               ##
######################################################

######################## read image ################################
# For one image

NAME = 'test'
# im = io.imread(f"./dataSample/{NAME}.jpg")
im = io.imread('/root/mnt/share/Voxceleb/train/id10841#5h5NmY820_M#001845#001980.mp4/0000000.png')
H, W = im.shape[:2]
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
# bb = (0, 0, H - 1, W - 1)
# coords = fa.get_landmarks(im)[0].astype(int)
# frame_landmarks = np.stack([frame_landmarks[:, 0].clip(0, H - 1), frame_landmarks[:, 1].clip(0, W - 1)], axis=1)
# (vertices, mesh_plotting, Ind, rotation_angle) = helpers.landmarks_3d_fitting(frame_landmarks,height,width)

(bb, coords) = helpers.get_landmarks(im)
print(f'bb: {bb}')
print(f'coords: {coords}')
(vertices, mesh_plotting, Ind, rotation_angle, ) = helpers.landmarks_3d_fitting(coords, H, W)

#np.savetxt('landmark_result_Yu.txt', coords)

# coords = mesh_plotting[Ind, :2].astype(int) + np.array([[W // 2, H // 2]])
################### output landmark coordinates ####################
for i in [0,16,36,43]:#np.arange(68):
    print("The " +  str(i) + "th landmark:")
    print(coords[i])
  

######################## visualization #############################
highlights = [16,36,43]
outImg = helpers.visualize_facial_landmarks(im, bb, coords, 1,highlights)
outImg_noBackground = helpers.visualize_facial_landmarks(im, bb, coords, 0,highlights)

plt.subplot(121), plt.imshow(outImg)
plt.subplot(122), plt.imshow(outImg_noBackground)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

fig = plt.figure()
outImg1 = helpers.visualize_facial_landmarks(im, bb, coords, 1,[])
plt.imshow(outImg1)

plt.show()

np.save(f'{NAME}_vertices.npy', vertices[Ind])
np.save(f'{NAME}_pose.npy', mesh_plotting[Ind])
fig.savefig(f'{NAME}_plot.png')
