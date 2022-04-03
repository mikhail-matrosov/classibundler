#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 23:19:32 2022

@author: miha
"""

import os
from functools import cached_property
from os.path import join as pjoin
import numpy as np


class Atlas():
    atlas_dir = os.path.dirname(__file__)

    @cached_property
    def data(self):
        fname = pjoin(self.atlas_dir, 'hcp842_80_atlas.npz')
        f = dict(np.load(fname, allow_pickle=1))
        self.label_names = f['label_names']
        self.hierarchy = f['hierarchy']
        # Has keys: streamlines, labels, label_names, hierarchy
        return f

    @property
    def streamlines(self):
        return self.data['streamlines']

    @property
    def labels(self):
        return self.data['labels']

    @cached_property
    def bundles(self):
        streamlines = self.streamlines
        labels = self.labels
        return {ln: streamlines[labels==i] for i, ln in enumerate(self.label_names)}

    @cached_property
    def centroids_data(self):
        fname = pjoin(self.atlas_dir, 'hcp842_80_centroids.npz')
        f = dict(np.load(fname, allow_pickle=1))
        self.label_names = f['label_names']
        self.hierarchy = f['hierarchy']
        # Has keys: streamlines, labels, label_names, hierarchy
        return f

    @property
    def centroids(self):
        return self.centroids_data['streamlines']

    @property
    def centroids_labels(self):
        return self.centroids_data['labels']

    @cached_property
    def centroids_bundles(self):
        streamlines = self.centroids
        labels = self.centroids_labels
        return {ln: streamlines[labels==i] for i, ln in enumerate(self.label_names)}

    @cached_property
    def label_names(self):
        return self.centroids_data['label_names']

    @cached_property
    def hierarchy(self):
        return self.centroids_data['hierarchy']


# Instantiate a standard atlas hcp842_80 with centroids
atlas = Atlas()
