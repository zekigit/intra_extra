import os
import mne
import sys

SUBJECT = sys.argv[1]
SUBJECTS_DIR = sys.argv[2]

# BEM segmentation
mne.bem.make_watershed_bem(SUBJECT, SUBJECTS_DIR, overwrite=True, volume='T1', show=True)

CONDUCTIVITY = (0.3, 0.006, 0.3)
MODEL = mne.make_bem_model(subject=SUBJECT, ico=6,
                           conductivity=CONDUCTIVITY,
                           subjects_dir=SUBJECTS_DIR)

# BEM  solution
BEM = mne.make_bem_solution(MODEL)
mne.write_bem_solution(SUBJECTS_DIR + '/%s/%s-bem-sol.fif' % (SUBJECT, SUBJECT), BEM)
