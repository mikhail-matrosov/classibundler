#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 02:36:23 2022

@author: miha
"""

import numpy as np
from os.path import join as pjoin
from drawing_utils import normalized
from features import features_unpacked


bundle_viewpoints = {
    'CST_L': (-1000, 0, 0),
    'CST_R': ( 1000, 0, 0),
    'CC_ForcepsMajor': (0, 0, 1000),
    'CC_ForcepsMinor': (0, 0, 1000),
    'CC_Mid': (-1000, 1000, 1000),
    'ILF_L': (-1000, 0, 0),
    'ILF_R': ( 1000, 0, 0),
    'IFOF_L': (0, 0, 1000),
    'IFOF_R': (0, 0, 1000),
    'UF_L': (-1200, 400, 1200),
    'UF_R': ( 1200, 400, 1200),
    'MCP': (0, 0, 1000),
    'AF_L': (-1500, -400, 1000),
    'AF_R': ( 1500, -400, 1000),
    'AST_L': (-200, 1000, 500),
    'AST_R': ( 200, 1000, 500),
    'FPT_L': (-1000, 0, 0),
    'FPT_R': ( 1000, 0, 0),
    'MdLF_L': (-1000, 0, 1000),
    'MdLF_R': ( 1000, 0, 1000),
    'PPT_L': (-1000, 0, 0),
    'PPT_R': ( 1000, 0, 0),
    'EMC_L': (-1000, 0, 0),
    'EMC_R': ( 1000, 0, 0)
}


def pyrdn(img, fac=4):
    h, w, d = img.shape
    lut = np.square(np.arange(256, dtype=np.float32))
    ds = lut[img].reshape((h//fac, fac, w//fac, fac, d)).mean(3).mean(1)
    return (np.sqrt(ds) + 0.5).astype(np.uint8)


def render_bundle(streamlines, bundle,
                  cam_pos=(-1000, 0, 0),
                  features=[(10, 20)],
                  fname=None,
                  size=512):
    from fury import actor, window
    from dipy.tracking.streamline import set_number_of_points

    bundle = set_number_of_points(bundle, 100)
    colors = np.zeros((100, 3), dtype=np.float32)
    colors[:] = (0, 1, 0)
    feature_colors = [
        (1, 0, 0),
        (.7, .7, 0),
        (0, .7, .7),
        (0, 0, 1),
        (1, 0, 1)
    ]
    for (l, h), col in zip(features, feature_colors):
        colors[l:h] = col

    scene = window.Scene()
    scene.SetBackground(1, 1, 1)
    if streamlines is not None and len(streamlines):
        scene.add(actor.streamtube(
            streamlines,
            colors=(0, 0, 0),
            linewidth=0.1,
            opacity=0.05
        ))
    for b in bundle:
        scene.add(actor.streamtube(
            [b],
            colors=colors,
            linewidth=0.1,
            opacity=0.5
        ))

    m = np.min([f.min(0) for f in bundle], 0)
    M = np.max([f.max(0) for f in bundle], 0)
    center = (m+M) / 2
    diameter = np.linalg.norm(m - M)
    distance = np.linalg.norm(center - cam_pos)
    fov = np.rad2deg(diameter / distance)

    # Direction arrow
    b0 = bundle[0]
    direction = normalized(b0[-1] - b0[len(b0) // 3 * 2])
    scene.add(actor.arrow([b0[-1] + direction * diameter * 0.05],
                          [direction],
                          colors=(1, 0, 0),
                          heights=diameter * 0.07))

    cam = scene.camera()
    cam.SetViewAngle(fov)
    cam.SetClippingRange(100, 10000)
    cam.SetFocalPoint(*center)
    cam.SetPosition(*cam_pos)

    x, y, z = np.abs(cam_pos)
    if z > x and z > y:
        cam.SetViewUp(0, -1, 0)
    else:
        cam.SetViewUp(0, 0, 1)

    if fname:
        # Render in higher resolution, downscale and save to file
        img = window.snapshot(scene, fname=None, size=(4*size, 4*size),
                              order_transparent=True, multi_samples=1)
        window.save_image(pyrdn(img), fname)
    else:
        # Interactive
        window.show(scene, size=(size, size), reset_camera=False)
        scene.camera_info()


def render_bundles(patient, output_dir, ghost_streamlines=True):
    # Render bundles
    for b, cam_pos in bundle_viewpoints.items():
        try:
            features = sorted({
                    f.slice for f in features_unpacked
                    if f.bundle==b and f.slice[1] - f.slice[0] < 70
                },
                key=lambda s: (s[0]-s[1], s[0])
            )

            render_bundle(patient.streamlines[::10] if ghost_streamlines else [],
                          patient.classified_bundles[b],
                          cam_pos=cam_pos,
                          features=features,
                          fname=pjoin(output_dir, f'{b}.jpg'))
        except:
            pass


'''
fprefix = '/media/miha/0c44a000-6bfa-4732-929b-f31bc6cf4011/miha/YandexDisk/MRI/Alexey'
patient_folder = fprefix + '/Patients/Healthy/KremnevaLA'
patient = Patient(patient_folder)
streamlines = patient.streamlines
bundle = patient.classified_bundles['UF_L']
render_bundle(patient.streamlines, patient.classified_bundles['EMC_L'])


patient = Atlas()
render_bundles(patient, fprefix + '/Bundles')
'''
