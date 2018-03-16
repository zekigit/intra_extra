import mne
import sys
import os.path as op

fs_subj = sys.argv[1]
subjects_dir = sys.argv[2]
study_path = sys.argv[3]
img_type = 'anony' if fs_subj.find('_an') > 0 else 'orig'
subj = fs_subj.strip('_an')


src = mne.setup_source_space(fs_subj, spacing='all', subjects_dir=subjects_dir)
mne.write_source_spaces(op.join(study_path, 'source_stim', subj, 'source_files', img_type, '%s-oct5-src.fif' % fs_subj),
                        src, overwrite=True)

