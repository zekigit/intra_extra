import mne
import sys
import os.path as op

subj = sys.argv[1]
subjects_dir = sys.argv[2]
study_path = sys.argv[3]

src = mne.setup_source_space(subj, spacing='oct5', subjects_dir=subjects_dir)
mne.write_source_spaces(op.join(study_path, 'source_stim', subj, 'source_files', '%s-oct5-src.fif' % subj), src, overwrite=True)

