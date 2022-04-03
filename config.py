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
group_names = ('Healthy', 'F3_MRT2')
nthreads = 4  # 1 for singleprocessing
