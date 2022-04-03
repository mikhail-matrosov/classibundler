#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 01:50:05 2022

@author: miha
"""

import os
from functools import cached_property
from os.path import join as pjoin, expanduser
from dipy.io.image import load_nifti
from dipy.io.streamline import load_trk, save_trk
import numpy as np
import processing as pr
from atlas import atlas


class Patient:
    dti_seq_name = 'data_s'
    mask_seq_name = 'brain_mask'
    b0_seq_name = '25_ep2d_diff_mddw_64_p2_s2_b0_pa'

    def __init__(self, folder):
        self.folder = expanduser(folder)

    @cached_property
    def nifty_dir(self):
        return pr.ensure_nifty(self.folder)

    @cached_property
    def proc_dir(self):
        output_dir = pjoin(self.folder, 'PROCESSED')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @cached_property
    def mask(self):
        mask_fname = pjoin(self.nifty_dir, self.mask_seq_name+'.nii.gz')
        if os.path.exists(mask_fname):
            # Load from file
            self._mask, self.affine = load_nifti(mask_fname)
            print(f'Loaded mask from {mask_fname}')
        else:
            b0_fname = pjoin(self.nifty_dir, self.b0_seq_name+'.nii.gz')
            self._mask, self.affine = pr.detect_brain_from_B0(b0_fname)

        return self._mask

    @cached_property
    def csa_peaks(self):
        fnprefix = pjoin(self.nifty_dir, self.dti_seq_name)
        data, self.affine, self.dti_img, gtab = pr.load_dti_data(fnprefix)
        return pr.get_csa_peaks(data, self.mask, gtab)

    @cached_property
    def streamlines(self):
        ''' Fiber tracking '''
        fname = pjoin(self.proc_dir, 'streamlines.trk')

        try:
            # Try loading from file
            self.sft = load_trk(fname, "same", bbox_valid_check=False)
            self.affine = self.sft.affine
            print(f'Loaded streamlines from {fname}')
            return self.sft.streamlines

        except FileNotFoundError:
            streamlines = pr.get_streamlines(self.csa_peaks, self.mask, self.affine)

            # Save streamlines
            from dipy.io.stateful_tractogram import Space, StatefulTractogram
            self.sft = StatefulTractogram(streamlines, self.dti_img, Space.RASMM)
            save_trk(self.sft, fname, bbox_valid_check=False)

            return streamlines

    @cached_property
    def regestered_streamlines(self):
        print('Registering patient to the Atlas...')
        from dipy.align.streamlinear import whole_brain_slr

        moved, self.transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
            atlas.streamlines, self.streamlines, x0='affine', verbose=True,
            progressive=True, rng=np.random.RandomState(1984))

        return moved

    @cached_property
    def streamlines_classification(self):
        fname = pjoin(self.proc_dir, 'streamlines_classification.npz')

        try:
            f = dict(np.load(fname))
            self.is_reversed = f['is_reversed']
            self.transform = f['transform']
            print(f'Streamlines classification is loaded from {fname}')
            return f['predictions']

        except FileNotFoundError:
            predictions, self.is_reversed = pr.classify_streamlines(
                self.regestered_streamlines, atlas.centroids, atlas.centroids_labels)

            np.savez(fname, predictions=predictions, is_reversed=self.is_reversed, transform=self.transform)
            return predictions

    @cached_property
    def classified_bundles(self):
        return pr.streamlines_as_dict(self.streamlines,
                                      self.streamlines_classification,
                                      self.is_reversed,
                                      atlas.label_names,
                                      atlas.centroids,
                                      atlas.centroids_labels)

    @cached_property
    def profiles_weights(self):
        fname = pjoin(self.proc_dir, 'profiles_weights.npz')

        try:
            return dict(np.load(fname))
            print(f'Loaded profiles weights from {fname}')
        except FileNotFoundError:
            weights = pr.get_profile_weights(
                self.classified_bundles, atlas.centroids, atlas.centroids_labels)
            np.savez(fname, **weights)
            return weights

    def profiles_metric(self, metric_name):
        fname = pjoin(self.proc_dir, f'profiles_{metric_name}.npz')

        try:
            profiles = dict(np.load(fname, allow_pickle=True))
            for k, v in profiles.items():
                profiles[k] = v.tolist()
            print(f'Loaded {metric_name} from {fname}')
            return profiles
        except FileNotFoundError:
            cbundles = self.classified_bundles  # A dict of already oriented streamlines
            weights = self.profiles_weights
            metric_fname = pjoin(self.nifty_dir, metric_name + '.nii.gz')

            profiles = pr.get_profile_metric(cbundles, weights, metric_fname)
            np.savez(fname, **profiles)
            return profiles

    @property
    def profiles_features(self):
        from features import extract_features
        patient_data = {
            'FA': self.profiles_metric('data_s_DKI_fa'),
            'RD': self.profiles_metric('data_s_DKI_rd'),
            'AD': self.profiles_metric('data_s_DKI_ad'),
            'MD': self.profiles_metric('data_s_DKI_md'),
        }
        return extract_features(patient_data)


class AtlasPatient(Patient):
    def __init__(self):
        self.folder = ''

    @property
    def streamlines(self):
        return atlas.streamlines

    @property
    def regestered_streamlines(self):
        return atlas.streamlines

    @cached_property
    def streamlines_classification(self):
        self.is_reversed = np.zeros(len(atlas.labels))
        return atlas.labels

atlas_patient = AtlasPatient()
