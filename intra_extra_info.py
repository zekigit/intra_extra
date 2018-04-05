import numpy as np
import platform

study_path_mac = '/Volumes/MAXTOR/'
study_path_ubu = '/home/eze/intra_extra/'


subj_dir_mac = '/Volumes/MAXTOR/freesurfer_subjects'
subj_dir_ubu = '/home/eze/fs_subjects'

study_path = study_path_ubu if platform.system() == 'Linux' else study_path_mac
subjects_dir = subj_dir_ubu if platform.system() == 'Linux' else subj_dir_mac

shared_folder = '/media/sf_share_vm/epochs'

subjects = ['S1', 'S2', 'S3', 'S4', 'S3']

subj_dig_mont = {'S1':  {'ch_names': ['1', '3'] + ['E' + str(i + 1) for i in range(256)] + ['2'],
                         'ch_types': ['fid'] + ['fid'] + ['eeg'] * 256 + ['fid'],
                         'dig_sel':  np.arange(259)},
                 'S2':  {'ch_names': ['1', '3'] + ['E' + str(i + 1) for i in range(256)] + ['2'],
                         'ch_types': ['fid'] + ['fid'] + ['eeg'] * 256 + ['fid'],
                         'dig_sel' : np.arange(259)},
                 'S3':  {'ch_names': ['E' + str(i + 1) for i in range(256)] + ['3', '2', '1'],
                         'ch_types': ['eeg'] * 256 + ['fid'] * 3,
                         'dig_sel' : np.concatenate((np.arange(256), [256, 257, 258]))},
                 'S4':  {'ch_names': ['E' + str(i + 1) for i in range(256)] + ['3', '2', '1'],
                         'ch_types': ['eeg'] * 256 + ['fid'] * 3,
                         'dig_sel' : np.concatenate((np.arange(256), [259, 260, 261]))},
                 'S5':  {'ch_names': ['1', '3'] + ['E' + str(i + 1) for i in range(256)] + ['2'],
                         'ch_types': ['fid'] + ['fid'] + ['eeg'] * 256 + ['fid'],
                         'dig_sel' : np.arange(259)}}


egi_outside_chans = ['E%i' % ch for ch in np.sort([241, 244, 248, 252, 253, 242, 245, 249, 254, 243, 246, 250, 255, 247, 251, 256, 73, 82,
                             91, 92, 102, 93, 103, 111, 104, 112, 120, 113, 121, 133, 122, 134, 145, 135, 146, 147,
                             156, 157, 165, 166, 167, 174, 175, 176, 187, 188, 189, 199, 200, 201, 208, 209, 216, 217,
                             218, 225, 227, 228, 229, 226, 230, 234, 238, 239, 235, 231, 232, 233, 236, 237, 240])]



