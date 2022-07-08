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

frame_size = 256
landmark_path = '../flame-fitting/output_landmark/fit_landmarks.npy'
landmark = np.load(landmark_path) * 1000


######################## visualization #############################
# plt.subplot(11)
# ax2 = plt.subplot(111)
# ax2.imshow(im2),ax2.set_title('Landmark Extraction on Raw Data', fontsize=16)
fig = plt.figure()
ax = fig.subplots()
ax.set_title('Landmarks Tracking', fontsize=16)
ax.set_ylabel('Pixel', fontsize=14)
ax.axis([-100, 100, -100, 100])
# landmarkCompare = 0 * im1.copy() + 255
# frame_landmarks_ed = mesh_
for (x, y) in landmark[:, 0:2]:
    x = np.int32(x)
    y = np.int32(y)
    ax.plot(x, y, 'go')

# fig = plt.figure()

# np.save(f'{NAME}_vertices.npy', vertices[Ind])
# np.save(f'{NAME}_pose.npy', mesh_plotting[Ind])
fig.savefig(f'flame_plot.png')
