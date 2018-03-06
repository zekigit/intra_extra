import numpy as np
import platform


study_path_mac = '/Volumes/MAXTOR/'
study_path_ubu = '/home/eze/intra_extra/'


subj_dir_mac = '/Volumes/MAXTOR/freesurfer_subjects'
subj_dir_ubu = '/home/eze/fs_subjects'

study_path = study_path_ubu if platform.system() == 'Linux' else study_path_mac
subjects_dir = subj_dir_ubu if platform.system() == 'Linux' else subj_dir_mac


subjects = ['S1', 'S2', 'S3', 'S4', 'S3']

subj_dig_mont = {'S1':  {'ch_names': ['1', '3'] + ['E' + str(i + 1) for i in range(256)] + ['2'],
                         'ch_types': ['fid'] + ['fid'] + ['eeg'] * 256 + ['fid'],
                         'dig_sel': np.arange(259)},
                 'S2':  {'ch_names': [],
                         'ch_types': [],
                         'dig_sel' : []},
                 'S3':  {'ch_names': ['E' + str(i + 1) for i in range(256)] + ['3', '2', '1'],
                         'ch_types': ['eeg'] * 256 + ['fid'] * 3,
                         'dig_sel' : np.concatenate((np.arange(256), [256, 257, 258]))},
                 'S4':  {'ch_names': ['E' + str(i + 1) for i in range(256)] + ['3', '2', '1'],
                         'ch_types': ['eeg'] * 256 + ['fid'] * 3,
                         'dig_sel' : np.concatenate((np.arange(256), [259, 260, 261]))},
                 'S5':  {'ch_names': ['1', '3'] + ['E' + str(i + 1) for i in range(256)] + ['2'],
                         'ch_types': ['fid'] + ['fid'] + ['eeg'] * 256 + ['fid'],
                         'dig_sel' : np.arange(259)}}
