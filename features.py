#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 01:39:18 2022

@author: miha
"""

features = (
    # (list_of_metrics, list_of_bundles, *list_of_slices)
    ('FA AD MD', 'CST_L CST_R', (30, 50)),
    ('FA AD', 'CC_ForcepsMajor', (30, 70)),
    ('FA RD AD MD', 'CC_ForcepsMinor', (20, 40), (40, 60), (60, 80)),
    ('RD MD', 'CC_Mid', (5, 35), (65, 95)),
    ('FA', 'ILF_L ILF_R', (30, 45), (70, 85)),
    ('FA', 'IFOF_L IFOF_R', (40, 60)),
    ('FA', 'UF_L UF_R', (80, 92)),
    ('FA', 'MCP', (5, 20)),
    ('FA RD AD MD', 'CST_L CST_R CC_ForcepsMajor CC_ForcepsMinor CC_Mid', (5, 95)),
    ('FA', '''
     ILF_L ILF_R IFOF_L IFOF_R UF_L UF_R MCP
     AF_L AF_R AST_L AST_R FPT_L FPT_R MdLF_L MdLF_R PPT_L PPT_R EMC_L EMC_R
     ''', (5, 95)),
)


##############################################################################

import numpy as np
from collections import namedtuple
Feature = namedtuple('Feature', 'name metric bundle slice')

features_unpacked = tuple(
    Feature(f'{m}_{b}_{s[0]}_{s[1]}', m, b, s)
    for metrics, bundles, *slices in features
    for m in metrics.split()
    for b in bundles.split()
    for s in slices
)


def extract_features(patient_data):
    '''
    patient_data is a dict:
        {
          metric: {
            bundle_name: {
              mean: [],
              std: []
            }
          }
        }

    Example:
        patient = Patient('path/to/patient/data')
        features = patient.profiles_features

    is equivalent to:
        patient_data = {
            'FA': patient.profiles_metric('data_s_DKI_fa'),
            'RD': patient.profiles_metric('data_s_DKI_rd'),
            'AD': patient.profiles_metric('data_s_DKI_ad'),
            'MD': patient.profiles_metric('data_s_DKI_md'),
        }
        features = extract_features(patient_data)

    '''
    features = {}

    for f in features_unpacked:
        try:
            d = patient_data[f.metric][f.bundle]
            a, b = f.slice
            features[f.name] = (
                np.mean(d['mean'][a:b+1]),  # Value = mean
                np.mean(d['std'][a:b+1]),   # std
            )
        except:
            features[f.name] = (0, 0)

    return features
