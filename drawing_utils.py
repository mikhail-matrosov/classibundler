#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 02:39:42 2022

@author: miha
"""

import numpy as np


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
