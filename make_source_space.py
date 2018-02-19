import mne
import sys

subj = sys.argv[0]
src = mne.setup_source_space(subj, spacing='oct5')
mne.write_source_spaces('%s-oct5-src.fif' % subj, src, overwrite=True)
