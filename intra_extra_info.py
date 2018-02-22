import numpy as np

study_path = '/Volumes/MAXTOR/'
subjects = ['S1', 'S2', 'S3', 'S4']

subjects_conds = {'S1': ['BLINK', 'NREM', 'NREM_2', 'NREM_3', 'R\'2-3_W_5ma_nn']}
subjects_dir = '/Volumes/MAXTOR/freesurfer_subjects'

bad_chans = {'S1': ['E61', 'E67', 'E68', 'E72', 'E92', 'E165', 'E174', 'E187'],
             'S2': [],
             'S3': [],
             'S4': [],
             'S5': ['E18', 'E25', 'E26', 'E61', 'E67', 'E68', 'E72', 'E92', 'E93', 'E94', 'E101', 'E102', 'E120','E145', 'E165', 'E174',
                    'E187', 'E199', 'E208', 'E251']}


subj_dig_mont = {'S1':  {'ch_names': ['1', '3'] + ['E' + str(i + 1) for i in range(256)] + ['2'],
                         'ch_types': ['fid'] + ['fid'] + ['eeg'] * 256 + ['fid'],
                         'dig_sel': np.arange(259)},
                 'S2':  {'ch_names': [],
                         'ch_types': [],
                         'dig_sel' : []},
                 'S3':  {'ch_names': [],
                         'ch_types': [],
                         'dig_sel' : []},
                 'S4':  {'ch_names': ['E' + str(i + 1) for i in range(256)] + ['3', '2', '1'],
                         'ch_types': ['eeg'] * 256 + ['fid'] * 3,
                         'dig_sel' : np.concatenate((np.arange(256), [259, 260, 261]))},
                 'S5':  {'ch_names': ['1', '3'] + ['E' + str(i + 1) for i in range(256)] + ['2'],
                         'ch_types': ['fid'] + ['fid'] + ['eeg'] * 256 + ['fid'],
                         'dig_sel' : np.arange(259)}}
