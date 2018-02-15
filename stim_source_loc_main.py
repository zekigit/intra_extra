import mne
import numpy as np
import os.path as op
import glob
import pandas as pd
from intra_extra_info import study_path, subjects, subjects_conds
from intra_extra_fx import load_locs, create_seeg_info, plot_locs, load_eeg
pd.set_option('display.expand_frame_repr', False)


subj = subjects[0]
epo_path = op.join(study_path, 'source_stim', 'epochs')
conds = glob.glob(epo_path + '/*.edf')

cond_fname = conds[0]


# Load EEG
def load_data(subj, cond_fname):
    eeg_raw = mne.io.read_raw_edf(cond_fname, preload=True)
    eeg_raw._data /= 1e6
    return eeg_raw

def create_epochs(eeg_raw):
    pass



eeg_raw = load_data(subj, cond_fname)

eeg_raw.plot()



