#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 21:43:08 2022

@author: miha
"""

import numpy as np
from dipy.io.image import load_nifti
from sklearn.mixture import GaussianMixture
from skimage.segmentation import flood
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import cv2

#fname = '/home/miha/mri/F3/ByvshevVV/__t2_spc_da-fl_sag_p2_iso_20190717133506_16.nii.gz'
fname = ('/media/miha/0c44a000-6bfa-4732-929b-f31bc6cf4011/miha/YandexDisk/MRI/Alexey/Patients'
         '/MS/BardinovaVV/_t2_space_dark-fluid_0.9_iso_tr7000_20210927175839_9.nii.gz')
data, affine = load_nifti(fname)
data = data * np.float32(1 / data.max())
sum_mask = np.zeros(data.shape, dtype=np.uint8)

# Supress
data -= gaussian(data < 0.05, 1).astype(np.float32)


#roi = data[120, 229:264, 278:324]
#m=-1
#roi = data[120, 229-m:264+m, 278-m+5:324+m]
#plt.imshow(roi)
#plt.imshow(roi)

#gm = GaussianMixture(n_components=2).fit(roi.reshape(-1, 1))
#
#bins = 60
#gmm_x = np.linspace(roi.min(), roi.max(), bins)
#gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1,1)))
#
#plt.hist(roi.flatten(), bins=bins, density=True)
#plt.plot(gmm_x, gmm_y, color='orange')
#
#mask = np.zeros(data.shape, dtype=np.uint8)
#mouseX, mouseY = 0, 0

#    P = gmm.predict(roi.reshape(-1, 1)).reshape(roi.shape)
#    layer_mask = (P==label).astype(np.uint8)
#    contours, h = cv2.findContours(layer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#    # Select the longest contour
#    contours = sorted(contours, key=len, reverse=True)
#    # Find a single contour containg the mouse cursor
#    for contour in contours:
#        if len(contour) < radius and len(contour) < 10:
#            break
#        layer_mask.fill(0)
#        cv2.drawContours(layer_mask, [contour], -1, color=1, thickness=-1)
#        if layer_mask[radius, radius]:  # Check cursor
#            return layer_mask


def detect_lession(z, y, x, radius, grow=3):
    # Construct a gaussian model
    roi = data[z, y-radius:y+radius, x-radius:x+radius]
    gmm = GaussianMixture(n_components=3).fit(roi.reshape(-1, 1))

    # Find a threshold
    label = np.argmax(gmm.means_.flatten())
    gmm_x = np.linspace(roi.min(), roi.max(), 1000)
    threshold = gmm_x[np.argmax(gmm.predict(gmm_x.reshape(-1, 1)) == label)]

    # Flood fill to find a connected region
    r = grow * radius
    print(x, y, z, r, data[z, y, x])
    mask = flood(data[max(z-r, 0):z+r, max(y-r, 0):y+r, max(x-r, 0):x+r] >= threshold,
                 tuple(r if c>r else c for c in [z, y, x]))

    # Crop
    zl, zh = np.where(mask.any(2).any(1))[0][[0, -1]]
    yl, yh = np.where(mask.any(2).any(0))[0][[0, -1]]
    xl, xh = np.where(mask.any(1).any(0))[0][[0, -1]]

    return mask[zl:zh+1, yl:yh+1, xl:xh+1], (max(z-r, 0) + zl,
                                             max(y-r, 0) + yl,
                                             max(x-r, 0) + xl)


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    mouseX, mouseY = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mask, (z0, y0, x0) = detect_lession(z, y, x, radius)
        d, h, w = mask.shape
        sum_mask[z0:z0+d, y0:y0+h, x0:x0+w] += mask


img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

z = 120
radius = 30

viewport = None
#overlay = None


while(1):
    img = data[z]
    viewport = np.ascontiguousarray(img).astype(np.float32)
#    if overlay is None:
#        overlay = np.zeros_like(viewport)
#    cv2.circle(viewport, (mouseX, mouseY), radius, (1,), 1)
    cv2.rectangle(viewport, (mouseX-radius, mouseY-radius), (mouseX+radius, mouseY+radius), (1,), 1)

    cv2.imshow('image', viewport + sum_mask[z] * 0.3)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
    if k == ord('a'):
        z = np.clip(z+1, 0, data.shape[0]-1)
    if k == ord('z'):
        z = np.clip(z-1, 0, data.shape[0]-1)
    if k == ord('x'):
        radius = max(radius-1, 1)
    if k == ord('s'):
        radius += 1
    if k == ord('c'):
        sum_mask.fill(0)

    if k != 255:
        print(k)

cv2.destroyAllWindows()
