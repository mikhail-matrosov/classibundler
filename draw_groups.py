#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 01:27:32 2021

@author: miha
"""

import numpy as np
import os
from os.path import join as pjoin
from scipy.stats import ttest_ind_from_stats

from processor import Patient
from config import mri_data_dir, group_names, output_dir

import matplotlib
matplotlib.use('Agg')  # File rendering
import matplotlib.pyplot as plt


os.makedirs(output_dir, exist_ok=True)

# Collect profiles for every patient in each group
profiles_dict = {k: {} for k in ['FA', 'RD', 'AD', 'MD']}

colors = ['blue', 'red', 'green', 'black']


def profiles_std(means, stds):
    N = len(means)
    mean = np.mean(means, 0)
    std2 = np.mean(np.square(stds), 0) + np.sum(np.square(means), 0) - N * mean**2
    return np.sqrt(std2 / N)


def means_std(means, stds):
    N = len(means)
    mean = np.mean(means)
    std2 = np.mean(np.square(stds)) + np.sum(np.square(means)) - N * mean**2
    return np.sqrt(std2 / N)


for gn in group_names:
    group = []
    gdir = pjoin(mri_data_dir, gn)
    patients = sorted(os.listdir(gdir))

    for p in patients:
        if '!' in p:
            print('IGNORING', gn, p)
        else:
            print(gn, p)
            patient = Patient(pjoin(gdir, p))
            group.append({
                'FA': patient.profiles_metric('data_s_DKI_fa'),
                'RD': patient.profiles_metric('data_s_DKI_rd'),
                'AD': patient.profiles_metric('data_s_DKI_ad'),
                'MD': patient.profiles_metric('data_s_DKI_md'),
            })

    label_names = patient.atlas_centroids['label_names']

    for k, d in profiles_dict.items():
        d[gn] = {
            ln: {
                'profiles': [p[k][ln]['mean'] for p in group if ln in p[k]],
                'mean': np.mean([p[k][ln]['mean'] for p in group if ln in p[k]], 0),
                'std': profiles_std([p[k][ln]['mean'] for p in group if ln in p[k]],
                                    [p[k][ln]['std']  for p in group if ln in p[k]]),
                'profiles_mean': np.mean([p[k][ln]['mean'].mean() for p in group if ln in p[k]]),
                'profiles_mean_std': means_std([p[k][ln]['mean'].mean() for p in group if ln in p[k]],
                                               [p[k][ln]['profiles'].mean(1).std() for p in group if ln in p[k]])
            }
            for ln in label_names
        }

# Calculate pvalues
gn1, gn2 = group_names[:2]
for k, d in profiles_dict.items():
    for ln in d[gn2]:
        if ln and d[gn1][ln]['profiles'] and d[gn2][ln]['profiles']:
            n1 = len(d[gn1][ln]['profiles'])
            n2 = len(d[gn2][ln]['profiles'])
            d[gn2][ln]['pvalue'] = [
                ttest_ind_from_stats(m1, s1, n1, m2, s2, n2).pvalue
                for m1, s1, m2, s2 in zip(
                    d[gn1][ln]['mean'],
                    d[gn1][ln]['std'],
                    d[gn2][ln]['mean'],
                    d[gn2][ln]['std']
                )
            ]


X = np.linspace(0, 100, 100)
print('Plot 1...')

# FOR ALEXEY
bundle_names = 'CST_L CST_R CC_ForcepsMajor CC_ForcepsMinor CC_Mid'.split()
nrows, ncols = len(profiles_dict), len(bundle_names)

fig = plt.figure(figsize=(30, 20))
fig.patch.set_alpha(1.0)
for r, p in enumerate(profiles_dict):
    for c, b in enumerate(bundle_names):
        index = ncols*r + c
        for gn, col in zip(group_names, colors):
            profs = profiles_dict[p][gn]
            if b in profs:
                ax = plt.subplot(nrows, ncols, index+1)
                ax.set_title(f'{p} in {b}')
                if p in ('FA',):
                    plt.ylim(0, 1)
#                if p == 'RD':
#                    plt.ylim(0, 1.5)
                profiles = profs[b]['profiles']
                N = len(profiles)
                mean = profs[b]['mean']
                std = profs[b]['std']
                pmean = profs[b]['profiles_mean']
                pstd = profs[b]['profiles_mean_std']
                # All tracts
                for profile in profiles:
                    plt.plot(X, profile, color=col, alpha=0.1)
                plt.plot(X, mean, color=col, label=f'{gn} (N={N}, {pmean:.2f}±{pstd:.2f})')
                plt.fill_between(X, mean-std, mean+std, color=col, alpha=0.2)
                leg = plt.legend(loc='upper left')
                plt.grid(b=True)

                # P-values
                if 'pvalue' in profs[b]:
                    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
                    color = 'tab:green'
#                    ax2.set_ylabel('pvalue', color=color)
                    ax2.semilogy(X, profs[b]['pvalue'], color=color, label='P-values')
                    ax2.tick_params(axis='y', labelcolor=color)
                    ax2.set_ylim(1, 1e-6)  # ax2.get_ylim()[0])
                    plt.legend(loc='upper right')


fname = pjoin(output_dir, 'profiles_plot_1')
# plt.savefig(fname + '.pdf')
# plt.savefig(fname + '.svg')
plt.savefig(fname + '.png', dpi=150, bbox_inches='tight')



print('Plot 2...')

fa_profiles = profiles_dict['FA']

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
        for gn, col in zip(group_names, colors):
            profs = profiles_dict['FA'][gn]
            if b in profs:
                ax = plt.subplot(nrows, ncols, index+1)
                ax.set_title(f'FA in {b}')
                plt.ylim(0, 1)
                profiles = profs[b]['profiles']
                N = len(profiles)
                mean = profs[b]['mean']
                std = profs[b]['std']
                pmean = profs[b]['profiles_mean']
                pstd = profs[b]['profiles_mean_std']
                # All tracts
                for profile in profiles:
                    plt.plot(X, profile, color=col, alpha=0.1)
                plt.plot(X, mean, color=col, label=f'{gn} (N={N}, {pmean:.2f}±{pstd:.2f}))')
                plt.fill_between(X, mean-std, mean+std, color=col, alpha=0.2)
                leg = plt.legend(loc='upper left')
                plt.grid(b=True)

                # P-values
                if 'pvalue' in profs[b]:
                    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
                    color = 'tab:green'
#                    ax2.set_ylabel('pvalue', color=color)
                    ax2.semilogy(X, profs[b]['pvalue'], color=color, label='P-values')
                    ax2.tick_params(axis='y', labelcolor=color)
                    ax2.set_ylim(1, 1e-6)  # ax2.get_ylim()[0])
                    plt.legend(loc='upper right')


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
        for gn, col in zip(group_names, colors):
            profs = profiles_dict['FA'][gn]
            if b in profs:
                ax = plt.subplot(nrows, ncols, index+1)
                ax.set_title(f'FA in {b}')
                plt.ylim(0, 1)
                profiles = profs[b]['profiles']
                N = len(profiles)
                mean = profs[b]['mean']
                std = profs[b]['std']
                pmean = profs[b]['profiles_mean']
                pstd = profs[b]['profiles_mean_std']
                # All tracts
                for profile in profiles:
                    plt.plot(X, profile, color=col, alpha=0.1)
                plt.plot(X, mean, color=col, label=f'{gn} (N={N}, {pmean:.2f}±{pstd:.2f}))')
                plt.fill_between(X, mean-std, mean+std, color=col, alpha=0.2)
                leg = plt.legend(loc='upper left')
                plt.grid(b=True)

                # P-values
                if 'pvalue' in profs[b]:
                    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
                    color = 'tab:green'
#                    ax2.set_ylabel('pvalue', color=color)
                    ax2.semilogy(X, profs[b]['pvalue'], color=color, label='P-values')
                    ax2.tick_params(axis='y', labelcolor=color)
                    ax2.set_ylim(1, 1e-6)  # ax2.get_ylim()[0])
                    plt.legend(loc='upper right')


fname = pjoin(output_dir, 'profiles_plot_3')
#plt.savefig(fname + '.pdf')
# plt.savefig(fname + '.svg')
plt.savefig(fname + '.png', dpi=150, bbox_inches='tight')


