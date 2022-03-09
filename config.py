'''
P=/media/miha/0c44a000-6bfa-4732-929b-f31bc6cf4011/miha/YandexDisk/MRI/Alexey/Patients

# Single patients:
python draw_profiles.py $P/Healthy/KremnevaLA $P/Healthy/KudryashovMI

# Each patient in a dir:
python draw_profiles.py -m $P/Healthy
python draw_profiles.py -m $P/ALS
python draw_profiles.py -m $P/MS
python draw_profiles.py -m $P/Fazekas3/Others
python draw_profiles.py -m $P/Fazekas3/Without_lacunes
python draw_profiles.py -m $P/MCAstroke/Left
python draw_profiles.py -m $P/MCAstroke/Right
python draw_profiles.py -m $P/F3_MRT2

# Draw groups and recalculate Healthy reference
python draw_groups.py
'''

mri_dir = '/media/miha/0c44a000-6bfa-4732-929b-f31bc6cf4011/miha/YandexDisk/MRI/Alexey/Patients'
output_dir = '/media/miha/0c44a000-6bfa-4732-929b-f31bc6cf4011/miha/YandexDisk/MRI/Alexey/Plots'

profiles_slice = slice(5, 95)  # slice(None) - all
draw_healthy = True  # For each single patient
render_whole_brain_ghost = True
group_names = ('Healthy', 'ALS')

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




##############################################################################

from collections import namedtuple
Feature = namedtuple('Feature', 'name metric bundle slice')

features_unpacked = tuple(
    Feature(f'{m}_{b}_{s[0]}_{s[1]}', m, b, s)
    for metrics, bundles, *slices in features
    for m in metrics.split()
    for b in bundles.split()
    for s in slices
)

