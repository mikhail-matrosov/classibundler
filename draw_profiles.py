#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    python3 draw_profiles.py path/to/patient1 path/to/patient2 path/to/patient3
    python3 draw_profiles.py -m path/to/group1 path/to/group2 path/to/group3

Created on Mon Dec 13 00:05:02 2021

@author: miha
"""

import numpy as np
import os
from os.path import join as pjoin
import sys

from processor import Patient
import config

import matplotlib
matplotlib.use('Agg')  # File rendering
import matplotlib.pyplot as plt


COLOR_XY2RGB_MTX = np.array([
    [ 1.,   0.,    0.5],
    [-0.5,  0.866, 0.5],
    [-0.5, -0.866, 0.5]
])


def normalized(v):
    return v / np.linalg.norm(v)


def colorize_bundle(bundle):
    main_direction = normalized(np.mean([f[-1] - f[0] for f in bundle], axis=0))
    X = normalized(np.cross(main_direction, [0, 0, 1]))
    Y = np.cross(main_direction, X)
    M = np.array([X, Y, main_direction])
    xyz = np.array([f[len(f)//2] for f in bundle]) @ M.T  # Project onto a cross-section plane
    xyz -= xyz.mean(0)
    xyz /= np.sqrt(np.square(xyz).mean(0))  # Normalize by sigma
    xyz[:,2] = 1
    return np.clip(xyz @ COLOR_XY2RGB_MTX.T, 0, 1)


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
    if streamlines:
        scene.add(actor.streamtube(
            streamlines[::10],
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

    # Add arrow
    direction = normalized(bundle[0][-1] - bundle[0][-2])
    scene.add(actor.arrow([bundle[0][-1] + direction * diameter * 0.05],
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

'''
patient_folder = '/media/miha/0c44a000-6bfa-4732-929b-f31bc6cf4011/miha/YandexDisk/MRI/Alexey/Patients/Healthy/KremnevaLA'
patient = Patient(patient_folder)
streamlines = patient.streamlines
bundle = patient.classified_bundles['UF_L']
render_bundle(patient.streamlines, patient.classified_bundles['EMC_L'])
'''


def draw_profiles(patient_folder):
    patient = Patient(patient_folder)
    fa_profiles = patient.profiles_metric('data_s_DKI_fa')
    profiles_dict = {
        'FA': fa_profiles,
        'RD': patient.profiles_metric('data_s_DKI_rd'),
        'AD': patient.profiles_metric('data_s_DKI_ad'),
        'MD': patient.profiles_metric('data_s_DKI_md')
    }

    output_dir = pjoin(patient.folder, 'PICS')
    os.makedirs(output_dir, exist_ok=True)

    print('Plot 1...')

    # FOR ALEXEY
    bundle_names = 'CST_L CST_R CC_ForcepsMajor CC_ForcepsMinor CC_Mid'.split()

    nrows, ncols = len(profiles_dict), len(bundle_names)

    fig = plt.figure(figsize=(30, 20))
    fig.patch.set_alpha(1.0)
    for r, p in enumerate(profiles_dict):
        for c, b in enumerate(bundle_names):
            index = ncols*r + c
            if b in profiles_dict[p]:
                plt.subplot(nrows, ncols, index+1)
                if p in ('FA', ):
                    plt.ylim(0, 1)
#                if p == 'RD':
#                    plt.ylim(0, 1.5)
                profiles = profiles_dict[p][b]['profiles']
                N = len(profiles)
                mean = profiles_dict[p][b]['mean']
                std = profiles_dict[p][b]['std']
                # All tracts
                for profile, color in zip(profiles, colorize_bundle(patient.classified_bundles[b])):
                    plt.plot(profile, alpha=0.1, color=color)
                plt.plot(mean, 'k', label=f'{p} {b} (N={N})')
                plt.plot(mean+std, 'k--')
                plt.plot(mean-std, 'k--')
                leg = plt.legend(
                    handlelength=0, handletextpad=0, fancybox=True, loc='upper left')
                for item in leg.legendHandles:
                    item.set_visible(False)
                plt.grid(b=True)

    fname = pjoin(output_dir, 'profiles_plot_1')
    # plt.savefig(fname + '.pdf')
    # plt.savefig(fname + '.svg')
    plt.savefig(fname + '.png', dpi=150, bbox_inches='tight')

    print('Plot 2...')

    # FOR MARIA (MAIN)
    bundle_names = [
        'CST_L CST_R ILF_L ILF_R'.split(),
        'IFOF_L IFOF_R UF_L UF_R'.split(),
        'CC_ForcepsMajor CC_Mid CC_ForcepsMinor MCP'.split(),
    ]

    nrows, ncols = len(bundle_names), len(bundle_names[0])

    fig = plt.figure(figsize=(30, 20))
    fig.patch.set_alpha(1.0)
    for r, bs in enumerate(bundle_names):
        for c, b in enumerate(bs):
            index = ncols*r + c
            if b in fa_profiles:
                plt.subplot(nrows, ncols, index+1)
                plt.ylim(0, 1)
                profiles = fa_profiles[b]['profiles']
                N = len(profiles)
                mean = fa_profiles[b]['mean']
                std = fa_profiles[b]['std']
                # All tracts
                for profile, color in zip(profiles, colorize_bundle(patient.classified_bundles[b])):
                    plt.plot(profile, alpha=0.1, color=color)
                plt.plot(mean, 'k', label=f'{b} (N={N})')
                plt.plot(mean+std, 'k--')
                plt.plot(mean-std, 'k--')
                leg = plt.legend(
                    handlelength=0, handletextpad=0, fancybox=True, loc='upper left')
                for item in leg.legendHandles:
                    item.set_visible(False)
                plt.grid(b=True)

    fname = pjoin(output_dir, 'profiles_plot_2')
    #plt.savefig(fname + '.pdf')
    # plt.savefig(fname + '.svg')
    plt.savefig(fname + '.png', dpi=150, bbox_inches='tight')

    print('Plot 3...')

    # FOR MARIA (ADDITIONAL)
    bundle_names = [
        'AF_L AF_R AST_L AST_R'.split(),
        'FPT_L FPT_R MdLF_L MdLF_R'.split(),
        'PPT_L PPT_R EMC_L EMC_R'.split(),
    ]

    nrows, ncols = len(bundle_names), len(bundle_names[0])

    fig = plt.figure(figsize=(30, 20))
    fig.patch.set_alpha(1.0)
    for r, bs in enumerate(bundle_names):
        for c, b in enumerate(bs):
            index = ncols*r + c
            if b in fa_profiles:
                plt.subplot(nrows, ncols, index+1)
                plt.ylim(0, 1)
                profiles = fa_profiles[b]['profiles']
                N = len(profiles)
                mean = fa_profiles[b]['mean']
                std = fa_profiles[b]['std']
                # All tracts
                for profile, color in zip(profiles, colorize_bundle(patient.classified_bundles[b])):
                    plt.plot(profile, alpha=0.1, color=color)
                plt.plot(mean, 'k', label=f'{b} (N={N})')
                plt.plot(mean+std, 'k--')
                plt.plot(mean-std, 'k--')
                leg = plt.legend(
                    handlelength=0, handletextpad=0, fancybox=True, loc='upper left')
                for item in leg.legendHandles:
                    item.set_visible(False)
                plt.grid(b=True)

    fname = pjoin(output_dir, 'profiles_plot_3')
    #plt.savefig(fname + '.pdf')
    # plt.savefig(fname + '.svg')
    plt.savefig(fname + '.png', dpi=150, bbox_inches='tight')

    # Render bundles
    for b, cam_pos in config.bundle_viewpoints.items():
        if b in fa_profiles:
            try:
                features = sorted({
                    f.slice for f in config.features_unpacked
                    if f.bundle==b and f.slice[1] - f.slice[0] < 70
                }, key=lambda s: (s[0]-s[1], s[0]))
                render_bundle(patient.streamlines,
                              patient.classified_bundles[b],
                              cam_pos=cam_pos,
                              features=features,
                              fname=pjoin(output_dir, f'{b}.jpg'))
#                render_bundle([],
#                              patient.classified_bundles[b],
#                              cam_pos=cam_pos,
#                              features=features,
#                              fname=pjoin(output_dir, f'{b}.jpg'))
            except:
                pass


print(sys.argv)
#folder = '/home/miha/mri/Healthy/GadzhievATO'
folder = sys.argv[-1]

from concurrent.futures import ProcessPoolExecutor
pool = ProcessPoolExecutor(4)

if len(sys.argv) > 2:
    if sys.argv[1] == '-m':
        # Process multiple patients from a folder
        paths = [pjoin(f, p) for f in sys.argv[2:] for p in sorted(os.listdir(f))]
        pool.map(draw_profiles, paths)
    else:
        pool.map(draw_profiles, sys.argv[1:])
else:
    print('''
Usage:
    python3 draw_profiles.py path/to/patient1 path/to/patient2 path/to/patient3
    python3 draw_profiles.py -m path/to/group1 path/to/group2 path/to/group3
'''
)














