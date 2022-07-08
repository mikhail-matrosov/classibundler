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

from os.path import join as pjoin

D = '/media/miha/0c44a000-6bfa-4732-929b-f31bc6cf4011/miha/YandexDisk/MRI/Alexey'
mri_dir = pjoin(D, 'Patients')
output_dir = pjoin(D, 'Plots')

profiles_slice = slice(5, 95)  # slice(None) - all
draw_healthy = True  # For each single patient
render_whole_brain_ghost = True
group_names = ('Healthy', 'MCAstroke/Left')
left_right_symmetries = {
    'MCAstroke/Left': 'MCAstroke/Right'
}
nthreads = 1  # 1 for singleprocessing

metrics = {
    'FA': 'data_s_DKI_fa',
    'RD': 'data_s_DKI_rd',
    'AD': 'data_s_DKI_ad',
    'MD': 'data_s_DKI_md',

    'IC': 'data_s_NODDI_IC',
    'ISO': 'data_s_NODDI_ISO',
    'ODI': 'data_s_NODDI_ODI',

    'Intra': 'data_s_smt_mc_intra',
    'Extramd': 'data_s_smt_mc_extramd',
    'Extratrans': 'data_s_smt_mc_extratrans',
}
