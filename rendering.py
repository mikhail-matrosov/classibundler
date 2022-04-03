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


def pyr_dn(img, fac=4):
    h, w, d = img.shape
    lut = np.square(np.arange(256, dtype=np.float32))
    ds = lut[img].reshape((h//fac, fac, w//fac, fac, d)).mean(3).mean(1)
    return (np.sqrt(ds) + 0.5).astype(np.uint8)


def beauty_bar_range(m, M, n_ticks=5):
    r = M - m
    tick = r / n_ticks
    scale = round(np.log10(tick))
    good_ticks = [.1, .15, .2, .25, .3, .4, .5, .75, 1]
    good_ticks = np.array(sorted(t * 10**s for s in range(scale-1, scale+2) for t in good_ticks))
    tick = good_ticks[good_ticks > tick][0]
    excessive = tick * n_ticks - r
    M = round((M + excessive / 2)/tick) * tick
    m = M - tick * n_ticks
    return m, M


def cmap_to_lut(cmap, min, max):
    '''
    Converts matplotlib.pyplot.ListedColormap to fury.lib.LookupTable
    '''
    import vtk
    from fury.lib import LookupTable

    table = (np.array(cmap.colors)*255 + 0.5).astype(np.uint8)

    vcolors = vtk.vtkUnsignedCharArray()
    vcolors.SetNumberOfComponents(4)
    vcolors.SetName("Colors")
    vcolors.SetNumberOfTuples(table.shape[0])
    for i, col in enumerate(table):
        vcolors.SetTuple4(i, *col, 0)

    lut = LookupTable()
    lut.SetTableRange((min, max))
    lut.SetTable(vcolors)

    return lut


def render_bundle(streamlines, bundle, *,
                  cam_pos=(-1000, 0, 0),
                  features=[],  # e.g. [(10, 20)]
                  fname=None,  # None - show in a window
                  colorscheme=None,  # None=green, 'RAINBOW', or plt colormaps
                  profiles=None,  # required if colorscheme is from plt, e.g. colorscheme='plasma'
                  metric_name = '', # E.g. 'FA'
                  arrow=False,
                  size=512):
    from fury import actor, window
    from dipy.tracking.streamline import set_number_of_points

    NP = 100 if profiles is None else profiles.shape[1]
    bundle = set_number_of_points(bundle, NP)

    img_scale = 4 if fname else 1

    # Colorize
    if colorscheme:
        if colorscheme == 'RAINBOW':
            pass
        else:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(colorscheme)
            m, M = np.min(profiles), np.max(profiles)
#            m, M = beauty_bar_range(m, M)
            colors = cmap((profiles - m) / (M - m))
    else:
        colors = np.zeros((len(bundle), NP, 3), dtype=np.float32)
        colors[:] = (0, 1, 0)

    # Mark features
    feature_colors = [
        (1, 0, 0),
        (.7, .7, 0),
        (0, .7, .7),
        (0, 0, 1),
        (1, 0, 1)
    ]
    for (l, h), col in zip(features, feature_colors):
        colors[:, l:h] = col

    scene = window.Scene()
    scene.SetBackground(1, 1, 1)
    if streamlines is not None and len(streamlines):
        scene.add(actor.streamtube(
            streamlines,
            colors=(0, 0, 0),
            linewidth=0.1,
            opacity=0.05
        ))
    for b, col in zip(bundle, colors):
        if colorscheme:
            scene.add(actor.line(
                [b],
                colors=col,
                linewidth=img_scale*2,
                depth_cue=True,
                opacity=0.5
            ))

        else:
            scene.add(actor.streamtube(
                [b],
                colors=col,
                linewidth=0.1,
                opacity=0.5
            ))

    if colorscheme:
        bar = actor.scalar_bar(cmap_to_lut(cmap, m, M), metric_name.ljust(8))
        bar.GetTitleTextProperty().SetColor(0.1, 0.1, 0.1)
        bar.GetLabelTextProperty().SetColor(0.1, 0.1, 0.1)
        bar.SetWidth(0.1)
        bar.SetBarRatio(0.2)
        scene.add(bar)

    # Detect bundle bounding box
    m = np.min([f.min(0) for f in bundle], 0)
    M = np.max([f.max(0) for f in bundle], 0)
    center = (m+M) / 2
    diameter = np.linalg.norm(m - M)
    distance = np.linalg.norm(center - cam_pos)
    fov = np.rad2deg(diameter / distance)

    # Direction arrow
    if arrow:
        b0 = bundle[0]
        direction = normalized(b0[-1] - b0[len(b0) // 3 * 2])
        scene.add(actor.arrow([b0[-1] + direction * diameter * 0.05],
                              [direction],
                              colors=(1, 0, 0),
                              heights=diameter * 0.07))

    # Point camera at the bundle
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
        img = window.snapshot(scene, fname=None,
                              size=(img_scale*size, img_scale*size),
                              order_transparent=True,
                              multi_samples=1)
        window.save_image(pyr_dn(img, img_scale), fname)
    else:
        # Interactive
        window.show(scene, size=(img_scale*size, img_scale*size), reset_camera=False)
        scene.camera_info()


def render_bundles(patient, output_dir, ghost=True, fa=True):
    # Render bundles
    if fa:
        profiles = patient.profiles_metric('data_s_DKI_fa')

    for b, cam_pos in bundle_viewpoints.items():
        try:
            features = sorted({
                    f.slice for f in features_unpacked
                    if f.bundle==b and f.slice[1] - f.slice[0] < 70
                },
                key=lambda s: (s[0]-s[1], s[0])
            )

            # With ghost and arrow
            render_bundle(patient.streamlines[::10] if ghost else [],
                          patient.classified_bundles[b],
                          cam_pos=cam_pos,
                          features=features,
                          fname=pjoin(output_dir, f'{b}.jpg'),
                          arrow=1)

            if fa:
                # FA With a colormap
                render_bundle([],
                              patient.classified_bundles[b],
                              cam_pos=cam_pos,
                              colorscheme='inferno',
                              profiles=profiles[b]['profiles'],
                              metric_name='FA',
                              fname=pjoin(output_dir, f'{b}_FA.jpg'),
                              )
        except:
            pass


'''
fprefix = '/media/miha/0c44a000-6bfa-4732-929b-f31bc6cf4011/miha/YandexDisk/MRI/Alexey'
patient_folder = pjoin(fprefix, 'Patients/Healthy/KremnevaLA')
from patient import Patient
bundle_name = 'PPT_L'
cam_pos = bundle_viewpoints[bundle_name]

# With ghost brain
patient = Patient(patient_folder)
streamlines = patient.streamlines
bundle = patient.classified_bundles[bundle_name]
render_bundle(patient.streamlines, bundle, cam_pos=cam_pos, arrow=1)

# Colorized
patient = Patient(patient_folder)
bundle = patient.classified_bundles[bundle_name]
profiles = patient.profiles_metric('data_s_DKI_fa')[bundle_name]['profiles']
render_bundle([], bundle, cam_pos=cam_pos, colorscheme='inferno', profiles=profiles, metric_name='FA')

# Render all bundles in a patient
patient = Patient(patient_folder)
render_bundles(patient, pjoin(patient_folder, 'PICS'))

# Render atlas bundles
from patient import atlas_patient
render_bundles(atlas_patient, fprefix + '/Bundles', fa=0)
'''
