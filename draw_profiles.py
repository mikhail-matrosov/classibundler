#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    python3 draw_profiles.py path/to/patient1 path/to/patient2 path/to/patient3
    python3 draw_profiles.py -m path/to/group1 path/to/group2 path/to/group3

Created on Mon Dec 13 00:05:02 2021

@author: miha
"""

import os
from os.path import join as pjoin
import sys

from patient import Patient
from drawing_utils import colorize_bundle
import config

import matplotlib
matplotlib.use('Agg')  # File rendering
import matplotlib.pyplot as plt


draw_metrics = 'FA RD AD MD'.split()


def draw_profiles(patient_folder):
    patient = Patient(patient_folder)
    profiles_dict = {
        m: patient.profiles_metric(m)
        for m in draw_metrics
    }
    fa_profiles = profiles_dict.get('FA') or patient.profiles_metric('FA')

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
                for profile, color in zip(profiles, colorize_bundle(patient.bundles[b])):
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
                for profile, color in zip(profiles, colorize_bundle(patient.bundles[b])):
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
                for profile, color in zip(profiles, colorize_bundle(patient.bundles[b])):
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

    render_bundles(patient, output_dir)


print(sys.argv)

# Multiprocessing
if config.nthreads > 1:
    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(config.nthreads)
    pool_map = pool.map
else:
    pool_map = lambda *x: [*map(*x)]


import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('paths', nargs='*', help='list of paths')
parser.add_argument('-r', '--render', action='store_true', help='render bundles for each patient')
parser.add_argument('-m', '--multiple', action='store_true', help='process all patients in a directory')

args = parser.parse_args()

if args.render:
    from rendering import render_bundles
else:
    render_bundles = lambda *x: None  # Do nothing

if args.multiple:
    # Process multiple patients from each directory
    paths = [pjoin(f, p) for f in args.paths for p in sorted(os.listdir(f))]
    pool_map(draw_profiles, paths)
else:
    pool_map(draw_profiles, args.paths)

'''
Usage:
    python3 draw_profiles.py path/to/patient1 path/to/patient2 path/to/patient3
    python3 draw_profiles.py -m path/to/group1 path/to/group2 path/to/group3
'''














