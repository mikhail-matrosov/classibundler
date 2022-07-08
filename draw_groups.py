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

from processing import swap_left_right_dict_keys
from patient import Patient
from atlas import atlas
import config
import features

import matplotlib
matplotlib.use('Agg')  # File rendering
import matplotlib.pyplot as plt


draw_metrics = 'IC ISO ODI'.split()

colors = ['blue', 'red', 'green', 'black']


os.makedirs(config.output_dir, exist_ok=True)
required_metrics = set(draw_metrics) | features.required_metrics  # To load
# Collect profiles for every patient in each group
profiles_dict = {k: {g: {} for g in config.group_names} for k in required_metrics}


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


for gn in config.group_names:
    gdir = pjoin(config.mri_dir, gn)
    patients_data = []
    patients_names = sorted(p for p in os.listdir(gdir) if '!' not in p)

    for p in patients_names:
        patient = Patient(pjoin(gdir, p))
        patients_data.append({
            m: patient.profiles_metric(m)
            for m in required_metrics
        })

    # Load symmetric patients
    if gn in config.left_right_symmetries:
        gdir = pjoin(config.mri_dir, config.left_right_symmetries[gn])
        patients_names_sym = sorted(p for p in os.listdir(gdir) if '!' not in p)
        patients_names.extend(patients_names_sym)
        for p in patients_names_sym:
            patient = Patient(pjoin(gdir, p))
            patients_data.append({
                m: swap_left_right_dict_keys(patient.profiles_metric(m))
                for m in required_metrics
            })

    label_names = atlas.label_names

    # Calculate statistics for plots
    for m in required_metrics:
        d = profiles_dict[m]
        for ln in label_names:
            ps = [p[m][ln] for p in patients_data if ln in p[m]]

            d[gn][ln] = {
                'profiles': [p['mean'] for p in ps],
                'mean': np.mean([p['mean'] for p in ps], 0),
                'std': profiles_std([p['mean'] for p in ps],
                                    [p['std']  for p in ps]),
                'profiles_mean': np.mean([p['mean'].mean() for p in ps]),
                'profiles_mean_std': means_std([p['mean'].mean() for p in ps],
                                               [p['profiles'].mean(1).std() for p in ps])
            }

    # Calculate features
    feats = {
        pn: features.extract_features(pdata)
        for pn, pdata in zip(patients_names, patients_data)
    }
    # Write features to a file
    gn = gn.replace('/', '').replace('\\', '')
    with open(pjoin(config.output_dir, f'Features_{gn}.csv'), 'wt') as f:
        fns = [f.name for f in features.features_unpacked]  # Feature names
        header = ','.join([''] + [col for h in fns for col in (h, 'std')])
        rows = [f'{p},' + ','.join(f'{x}' for fn in fns for x in fs[fn])
                for p, fs in feats.items()]
        f.write('\n'.join([header, *rows]))


# Write healthy profiles for later use
try:
    import pickle
    with open(pjoin(config.output_dir, 'profiles_dict_healthy.pkl'), 'wb') as f:
        pickle.dump({k: d['Healthy'] for k, d in profiles_dict.items()}, f)
except KeyError:
    print('No healthy group: skipped writing pkl')


# Calculate pvalues
gn1, gn2 = config.group_names[:2]
for m in required_metrics:
    d = profiles_dict[m]
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


S = config.profiles_slice
X = np.linspace(0, 100, 100)[S]


def plot_1(with_rainbow, with_pvalue):
    ''' A PLOT FOR ALEXEY
    '''
    bundle_names = 'CST_L CST_R CC_ForcepsMajor CC_ForcepsMinor CC_Mid'.split()
    nrows, ncols = len(draw_metrics), len(bundle_names)

    fig = plt.figure(figsize=(30, 20))
    fig.patch.set_alpha(1.0)
    for r, p in enumerate(draw_metrics):
        for c, b in enumerate(bundle_names):
            index = ncols*r + c
            for gn, col in zip(config.group_names, colors):
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
                    if with_rainbow:
                        for profile in profiles:
                            plt.plot(X, profile[S], color=col, alpha=0.1)

                    plt.plot(X, mean[S], color=col, label=f'{gn} (N={N}, {pmean:.2f}±{pstd:.2f})')
                    plt.fill_between(X, mean[S]-std[S], mean[S]+std[S], color=col, alpha=0.2)
                    plt.legend(loc='upper left')
                    plt.grid(b=True)

                    # P-values
                    if with_pvalue and 'pvalue' in profs[b]:
                        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
                        color = 'tab:green'
    #                    ax2.set_ylabel('pvalue', color=color)
                        ax2.semilogy(X, profs[b]['pvalue'][S], color=color, label='P-values')
                        ax2.tick_params(axis='y', labelcolor=color)
                        ax2.set_ylim(1, 1e-6)  # ax2.get_ylim()[0])
                        plt.legend(loc='upper right')

    groups = '_'.join(config.group_names).replace('/','').replace('\\','')
    metrics = '_'.join(draw_metrics).replace('/','').replace('\\','')
    suffix = ('R' if with_rainbow else '') + ('P' if with_pvalue else '')
    fname = pjoin(config.output_dir,
                  f"profiles_{groups}_{metrics}{suffix and '_'+suffix}")
    # plt.savefig(fname + '.pdf')
    # plt.savefig(fname + '.svg')
    plt.savefig(fname + '.png', dpi=150, bbox_inches='tight')


print('Plot 1...')
plot_1(with_rainbow=0, with_pvalue=0)
plot_1(with_rainbow=1, with_pvalue=0)
plot_1(with_rainbow=0, with_pvalue=1)
plot_1(with_rainbow=1, with_pvalue=1)


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
        for gn, col in zip(config.group_names, colors):
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
                    plt.plot(X, profile[S], color=col, alpha=0.1)
                plt.plot(X, mean[S], color=col, label=f'{gn} (N={N}, {pmean:.2f}±{pstd:.2f}))')
                plt.fill_between(X, mean[S]-std[S], mean[S]+std[S], color=col, alpha=0.2)
                leg = plt.legend(loc='upper left')
                plt.grid(b=True)

                # P-values
                if 'pvalue' in profs[b]:
                    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
                    color = 'tab:green'
#                    ax2.set_ylabel('pvalue', color=color)
                    ax2.semilogy(X, profs[b]['pvalue'][S], color=color, label='P-values')
                    ax2.tick_params(axis='y', labelcolor=color)
                    ax2.set_ylim(1, 1e-6)  # ax2.get_ylim()[0])
                    plt.legend(loc='upper right')


gns = '_'.join(config.group_names).replace('/','').replace('\\','')
fname = pjoin(config.output_dir, f"profiles_{gns}_2")
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
        for gn, col in zip(config.group_names, colors):
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
                    plt.plot(X, profile[S], color=col, alpha=0.1)
                plt.plot(X, mean[S], color=col, label=f'{gn} (N={N}, {pmean:.2f}±{pstd:.2f}))')
                plt.fill_between(X, mean[S]-std[S], mean[S]+std[S], color=col, alpha=0.2)
                leg = plt.legend(loc='upper left')
                plt.grid(b=True)

                # P-values
                if 'pvalue' in profs[b]:
                    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
                    color = 'tab:green'
#                    ax2.set_ylabel('pvalue', color=color)
                    ax2.semilogy(X, profs[b]['pvalue'][S], color=color, label='P-values')
                    ax2.tick_params(axis='y', labelcolor=color)
                    ax2.set_ylim(1, 1e-6)  # ax2.get_ylim()[0])
                    plt.legend(loc='upper right')


fname = pjoin(config.output_dir, f"profiles_{gns}_3")
#plt.savefig(fname + '.pdf')
# plt.savefig(fname + '.svg')
plt.savefig(fname + '.png', dpi=150, bbox_inches='tight')


