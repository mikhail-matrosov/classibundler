#!/usr/bin/env python
# coding: utf-8
'''
Based on https://dipy.org/documentation/1.3.0./examples_built/tracking_introduction_eudx/#example-tracking-introduction-eudx

Usage example:
folder = '/home/miha/mri/Healthy/GadzhievATO'
p = Patient(folder)
pm = p.profiles_metric('data_s_DKI_fa')

Author: ktdfly
2021-12-01
'''

import numpy as np
from dipy.io.image import load_nifti
import os
from os.path import join as pjoin


def ensure_nifty(dicom_dir):
    output_dir = pjoin(dicom_dir, 'RESULT')
    os.makedirs(output_dir, exist_ok=True)

#    if not os.listdir(output_dir):
#        print('Generating nifty...')
#        import dicom2nifti
#        dicom2nifti.convert_directory(dicom_dir, output_dir, compression=True, reorient=True)

    return output_dir


def detect_brain_from_B0(b0_fname):
    print('Generated mask from B0')
    b0_data, affine = load_nifti(b0_fname)
    b0_data = np.squeeze(b0_data)

    if len(b0_data.shape) == 4:
        b0_data = b0_data.mean(3)

    from dipy.segment.mask import median_otsu
    b0_mask, mask = median_otsu(b0_data)  #, median_radius=4, numpass=4)

    # # Erode mask
    # import scipy
    # mask = scipy.ndimage.morphology.binary_erosion(mask, iterations=2)

    return mask, affine


def load_dti_data(fnprefix):
    '''
    Loads DTI sequence
    https://dipy.org/documentation/1.1.1./examples_built/brain_extraction_dwi/
    '''
    print('Loading DTI data...')
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs

    data, affine, img = load_nifti(fnprefix+'.nii.gz', return_img=True)
    bvals, bvecs = read_bvals_bvecs(fnprefix+'.bval', fnprefix+'.bvec')
    gtab = gradient_table(bvals, bvecs)
    return data, affine, img, gtab


def get_csa_peaks(dti_data, mask, gtab):
    print('Calculating CSA peaks...')
    # from dipy.reconst.csdeconv import auto_response_ssst
    from dipy.reconst.shm import CsaOdfModel
    from dipy.data import default_sphere
    from dipy.direction import peaks_from_model

    # response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
    csa_model = CsaOdfModel(gtab, sh_order=4)  # 6
    csa_peaks = peaks_from_model(csa_model, dti_data, default_sphere,
                                 relative_peak_threshold=.8,
                                 min_separation_angle=45,
                                 mask=mask)
    return csa_peaks


def get_streamlines(csa_peaks, mask, affine):
    print('Calculating streamlines...')
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    from dipy.tracking import utils
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.streamline import Streamlines

    stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, .1)
    seeds = utils.seeds_from_mask(mask, affine, density=[1, 1, 1])
    streamlines_generator = LocalTracking(
            csa_peaks, stopping_criterion, seeds,
            affine=affine, step_size=.5)  # return_all=False
    streamlines = Streamlines(streamlines_generator)
    # Filter out empty streamlines (required if return_all=True)
    from nibabel.streamlines.array_sequence import ArraySequence
    streamlines = ArraySequence([s for s in streamlines if len(s) > 1])

    return streamlines


def classify_streamlines(streamlines, centroids, clabels, threshold=10):
    print(f'Classifiying streamlines...')
    from classibundler import classify
    from dipy.tracking.streamline import set_number_of_points

    NF = len(centroids[0]) // 3
    fibers = np.reshape(set_number_of_points(streamlines, NF), (-1, NF*3))
    predictions, is_reversed = classify(fibers, centroids, clabels, threshold=threshold)
    return predictions, is_reversed


def streamlines_as_dict(streamlines, streamlines_classification, is_reversed, label_names, centroids, clabels):
    print('Calculating streamlines_as_dict...')
    from nibabel.streamlines.array_sequence import ArraySequence
    from dipy.tracking.streamline import orient_by_streamline

    cbundles = {
        l: ArraySequence([
            f[::-1] if r else f
            for f, p, r in zip(streamlines, streamlines_classification, is_reversed)
            if p == i
            ])
        for i, l in enumerate(label_names) if i
    }

    # Orienting
    for i, k in enumerate(label_names):
        if k in cbundles:
            standard = centroids[clabels==i][0].reshape((-1, 3))
            cbundles[k] = orient_by_streamline(cbundles[k], standard)

    return cbundles


def get_profile_weights(classified_bundles, centroids, clabels):
    print('Calculating profile weights...')
    from dipy.stats.analysis import gaussian_weights
    weights = {}

    for i, k in enumerate(classified_bundles):
        B = classified_bundles[k]
        if len(B):
            weights[k] = gaussian_weights(B)
            if i % 5 == 4:
                print(f'{i+1}/{len(classified_bundles)}\t{k}\tN={len(B)}')
        else:
            print(f'{i+1}/{len(classified_bundles)}\t{k} is empty')

    return weights


def get_bundle_profile(bundle, weights, volumetric, affine):
    from dipy.stats.analysis import afq_profile

    profiles = np.array([afq_profile(volumetric.squeeze(), [o], affine) for o in bundle])
    mean = np.sum(profiles * weights, 0)
    std = np.sqrt(np.average(np.square(profiles-mean), weights=weights, axis=0))
    return {
        'profiles': profiles,
        'mean': mean,
        'std': std
    }


def get_profile_metric(cbundles, profiles_weights, metric_fname):
    print(f'Calculating profile metric for {metric_fname}...')
    m, affine = load_nifti(metric_fname)
    return {k: get_bundle_profile(cbundles[k], profiles_weights[k], m, affine)
            for k, w in profiles_weights.items()}


def swap_left_right_dict_keys(d):
    def swap_lr(s):
        if s.endswith('_L'):
            return s[:-2] + '_R'
        if s.endswith('_R'):
            return s[:-2] + '_L'
        return s
    return {swap_lr(k): v for k, v in d.items()}

